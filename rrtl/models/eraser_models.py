import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence

from transformers import BertModel

from rrtl.models.encoders import (
    TokenLevelEncoder,
    SentenceLevelEncoder,
)

from rrtl.models.decoders import (
    Decoder,
)

from lpsmap import (
    Budget,
    Pair,
    Sequence,
    SequenceBudget,
    TorchFactorGraph,
    Xor,
    Or,
    AtMostOne,
)


class FullContextERASERModel(nn.Module):
    def __init__(self, args):
        super(FullContextERASERModel, self).__init__()
        self.args = args
        self.decoder = Decoder(args)
        if self.args.use_gold_rationale:
            self.linear = nn.Linear(self.args.encoder_hidden_size, 1)
            self.drop = nn.Dropout(self.args.dropout_rate)
    
    def forward(self, batch, mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self.decoder(input_ids, attention_mask, labels)
        loss = output.loss

        if self.args.use_gold_rationale:
            rationales = batch['rationales']
            token_logits = self.drop(self.linear(output.hidden_states))
            gold_rationale_loss = self.calc_rationale_loss(token_logits, rationales, attention_mask)
            loss += self.args.gamma * gold_rationale_loss
        return {
            'loss': loss,
            'logits': output.logits,
        }

    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')

    def calc_rationale_loss(self, logits, rationale, mask):
        active_loss_pos = (mask.view(-1) == 1)
        active_logits = logits.view(-1)[active_loss_pos]
        active_labels = rationale.float().view(-1)[active_loss_pos]
        loss = F.binary_cross_entropy_with_logits(active_logits, active_labels)
        return loss


class VIBERASERSentModel(nn.Module):
    def __init__(self, args):
        super(VIBERASERSentModel, self).__init__()
        self.args = args
        self.encoder = SentenceLevelEncoder(args) # model 1
        self.rep_to_logit_layer = nn.Sequential(nn.Linear(args.encoder_hidden_size * 2, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 1))
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = Decoder(args) # model 2
    
    def forward(self, batch, mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        sent_starts = batch['sent_starts']
        sent_ends = batch['sent_ends']
        sent_lengths = batch['sent_lengths']
        fix_positions = batch['fix_positions']
        # # gold_sent_positions = batch['gold_sent_mask']

        sent_reps, token_reps = self.encoder(input_ids, attention_mask, sent_starts, sent_ends)

        batch_size, num_tokens, _ = token_reps.size() # 1, 128
        sent_logits = self.rep_to_logit_layer(sent_reps)
        sent_logits = self.drop(sent_logits)
        sent_logits = sent_logits.squeeze(2)
        sent_mask = self.create_sent_mask(sent_lengths, batch['max_sent_length']).float()

        sent_z = self.sample_z(sent_logits, sent_mask, mode)   ## sampling 
        ### new attention mask
        ### fix_positions b/c we want the query to always be masked to 1
        token_z = self.convert_z_from_sentence_to_token_level(
            sent_z,
            sent_starts,
            sent_ends,
            fix_positions,
            attention_mask
        )

        output = self.decoder(
            input_ids,
            token_z,
            labels
        )

        kl_loss = self.calc_kl_loss(sent_logits, sent_mask)
        loss = output.loss + self.args.beta * kl_loss

        if self.args.use_gold_rationale:
            # this part makes sense
            gold_sent_positions = [] # TODO remove
            sent_rationale_loss = self.calc_sent_rationale_loss(sent_logits, gold_sent_positions, sent_mask)
            loss += self.args.gamma * sent_rationale_loss
        else:
            sent_rationale_loss = torch.zeros(loss.size()).to(loss.device)

        # if self.args.dataparallel:
        #     loss = loss.item()
        #     kl_loss = kl_loss.item()


        return {
            'loss': loss,
            'pred_loss': output.loss,
            'logits': output.logits,
            'kl_loss': kl_loss,
            'token_z': token_z,
            'sent_z': sent_z,
            'sent_rationale_loss': sent_rationale_loss,
        }


    def create_sent_mask(self, sent_lengths, max_sent_length):
        max_sent_length = max_sent_length if self.args.dataparallel else sent_lengths.max().item()
        return torch.arange(max_sent_length).expand(sent_lengths.size(0), max_sent_length).to(sent_lengths.device) < sent_lengths.unsqueeze(1)
    
    def convert_z_from_sentence_to_token_level(self, z, sent_starts, sent_ends, fix_positions, attention_mask):
        # print(f'z: {z}')
        device = attention_mask.device
        mask = []
        sent_token_lengths = sent_ends - sent_starts
        # print(f'sent_token_lengths: {sent_token_lengths}')
        # print(f'fix_positions: {fix_positions}')
        for z_, sent_token_lengths_, fix_positions_ in zip(z, sent_token_lengths, fix_positions):
            fixed = torch.ones(1, fix_positions_).to(device)
            mask_ = torch.repeat_interleave(z_, sent_token_lengths_, dim=0).unsqueeze(0)
            mask_ = torch.cat((fixed, mask_), dim=1).squeeze(0)
            last_pos = mask_.size(0)
            pad_length = max(0, self.args.max_length - last_pos)
            mask_ =  F.pad(mask_, pad=(0, pad_length), mode='constant', value=0)
            mask_[last_pos] = 1.0  # set the trailing [SEP] on
            mask.append(mask_)
        mask = torch.stack(mask)
        return mask

    def calc_kl_loss(self, logits, mask):
        p = Bernoulli(logits=logits)
        prior_probs = torch.ones(logits.size()).to(logits.device) * self.args.pi
        r = Bernoulli(probs=prior_probs)
        kl_loss = kl_divergence(p, r)
        kl_loss = kl_loss * mask.float()
        kl_loss = kl_loss.sum(dim=1) / (mask.sum(dim=1))
        return kl_loss.mean()
    
    def calc_sent_rationale_loss(self, sent_logits, gold_sent_positions, sent_mask):
        device = sent_logits.device
        rationale = gold_sent_positions
        print(f'rationale size: {rationale.shape}')
        print(f'sent_logits size: {sent_logits.shape}')
        # rationale = torch.zeros(sent_mask.size()).to(device).scatter(1, gold_sent_positions.unsqueeze(1), 1.0)
        # print(f'rationale: {rationale}')
        active_loss_pos = (sent_mask.view(-1) == 1)
        # print(f'active_los_pos: {active_loss_pos}')
        active_logits = sent_logits.view(-1)[active_loss_pos]
        # print(f'active_logits: {active_logits} ')
        # active_labels = rationale.float().view(-1)[active_loss_pos]      
        active_labels = active_loss_pos
        sent_rationale_loss = F.binary_cross_entropy_with_logits(active_logits, active_labels)
        return sent_rationale_loss

    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')

    def sample_z(self, sent_logits, sent_mask, mode):
        device = sent_logits.device
        batch_size = sent_logits.size(0)
        lengths = sent_mask.sum(dim=1)

        if mode == 'train':
            relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=sent_logits)
            z = relaxed_bernoulli_dist.rsample()  # sample with reparameterization
        elif mode == 'eval':
            z = torch.sigmoid(sent_logits)

        # mask out sentence-level paddings
        z = z.masked_fill(~sent_mask.bool(), 0.0)

        if mode == 'eval':
            active_top_k = (lengths * self.args.pi).ceil().long()

            # this is essentially sorting from large to small
            _, z_hard_inds = z.topk(z.size(-1), dim=-1)
            z_hard = torch.zeros(z.size()).long().to(device)

            # TODO: must be a better way than using a loop here
            for i in range(z.size(0)):
                subidx = z_hard_inds[i, :active_top_k[i]]
                z_hard[i, subidx] = 1.0
            z = z_hard
        return z


class VIBERASERTokenModel(nn.Module):
    def __init__(self, args):
        super(VIBERASERTokenModel, self).__init__()
        self.args = args
        self.encoder = TokenLevelEncoder(args)
        self.rep_to_logit_layer = nn.Sequential(nn.Linear(args.encoder_hidden_size, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 1))
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = Decoder(args)
    
    def forward(self, batch, mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        fix_positions = batch['fix_positions']

        token_reps = self.encoder(input_ids, attention_mask)

        batch_size, num_tokens, _ = token_reps.size()
        token_logits = self.rep_to_logit_layer(token_reps)
        token_logits = self.drop(token_logits)
        token_logits = token_logits.squeeze(2)

        token_z = self.sample_z(token_logits, attention_mask, fix_positions, mode)

        output = self.decoder(input_ids, token_z, labels)

        kl_loss = self.calc_kl_loss(token_logits, attention_mask)
        loss = output.loss + self.args.beta * kl_loss

        return {
            'loss': loss,
            'pred_loss': output.loss,
            'logits': output.logits,
            'kl_loss': kl_loss,
            'token_z': token_z,
        }
    
    def sample_z(self, logits, mask, fix_positions, mode):
        device = logits.device
        batch_size = logits.size(0)
        lengths = mask.sum(dim=1)

        fix_mask = (torch.arange(logits.size(1)).expand(batch_size, logits.size(1)).to(device) < fix_positions.unsqueeze(1)).long()

        if mode == 'train':
            relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=logits)
            z = relaxed_bernoulli_dist.rsample()
            z = z.masked_fill(fix_mask.bool(), 1.0)
        elif mode == 'eval':
            z = torch.sigmoid(logits)
        # mask out paddings
        z = z.masked_fill(~mask.bool(), 0.0)

        if mode == 'eval':
            z = z.masked_fill(fix_mask.bool(), 1.0)
            active_top_k = ((lengths - fix_positions.sum(-1)) * self.args.pi).ceil().long()

            # this is essentially sorting from large to small
            _, z_hard_inds = z.topk(z.size(-1), dim=-1)
            z_hard = torch.zeros(z.size()).long().to(device)

            # TODO: must be a better way than using a loop here
            for i in range(z.size(0)):
                subidx = z_hard_inds[i, :active_top_k[i]]
                z_hard[i, subidx] = 1.0
            z = z_hard
        return z

    def calc_kl_loss(self, logits, mask):
        p = Bernoulli(logits=logits)
        prior_probs = torch.ones(logits.size()).to(logits.device) * self.args.pi
        r = Bernoulli(probs=prior_probs)
        kl_loss = kl_divergence(p, r)
        kl_loss = kl_loss * mask.float()
        kl_loss = kl_loss.sum(dim=1) / (mask.sum(dim=1))
        return kl_loss.mean()

    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')


class SpectraERASERSentModel(nn.Module):
    def __init__(self, args):
        super(SpectraERASERSentModel, self).__init__()
        self.args = args
        self.encoder = SentenceLevelEncoder(args)
        self.question_rep_proj = nn.Linear(args.encoder_hidden_size * 2, 128)
        self.context_rep_proj = nn.Linear(args.encoder_hidden_size * 2, 128)
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = Decoder(args)

    def forward(self, batch, mode='train', intervene=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        device = input_ids.device
        batch_size = input_ids.size(0)

        labels = batch['labels']

        sent_starts = batch['sent_starts']
        sent_ends = batch['sent_ends']
        sent_lengths = batch['sent_lengths']
        fix_positions = batch['fix_positions']
        gold_sent_positions = batch['gold_sent_positions']

        sent_reps, _ = self.encoder(input_ids, attention_mask, sent_starts, sent_ends)

        # separate question and passage representations
        question_rep = sent_reps[:, 0:1]
        context_reps = sent_reps[:, 1:]

        question_rep = F.relu(self.question_rep_proj(question_rep))
        context_reps = F.relu(self.context_rep_proj(context_reps))

        sent_logits = (context_reps * question_rep).sum(dim=-1)

        sent_mask = self.create_sent_mask(sent_lengths, batch['max_sent_length']).float()
        sent_z = self.sample_z(sent_logits, sent_mask, mode)
        token_z = self.convert_z_from_sentence_to_token_level(
            sent_z,
            sent_starts[:, 1:],
            sent_ends[:, 1:],
            fix_positions,
            attention_mask
        )
        output = self.decoder(input_ids, token_z, labels)
        loss = output.loss
        
        sent_rationale_loss = torch.zeros(loss.size()).to(loss.device)

        return {
            'logits': output.logits,
            'loss': loss,
            'pred_loss': output.loss,
            'token_z': token_z,
            'sent_z': sent_z,
            'sent_rationale_loss': sent_rationale_loss,
        }

    def sample_z(self, sent_logits, sent_mask, mode):
        device = sent_logits.device
        batch_size = sent_logits.size(0)
        sent_lengths = sent_mask.sum(dim=-1).long()
        z = []
        for i in range(batch_size):
            if self.args.budget_ratio is not None:
                budget = torch.round(self.args.budget_ratio * sent_lengths[i])
            elif self.args.budget is not None:
                budget = self.args.budget
            scores = sent_logits[i].unsqueeze(0)
            scores[0, ~sent_mask[i].bool()] = -1000.0
            z_probs = self.matching_smap_budget(
                scores,
                budget=budget,
                temperature=self.args.temperature,
                max_iter=self.args.solver_iter
            )
            z.append(z_probs)

        z = torch.cat(z, dim=0)
        return z

    def matching_smap_budget(self, scores, max_iter=100, temperature=1.0, init=True, budget=None):
        """
        M:Budget strategy for matchings extraction
        """
        m, n = scores.shape
        fg = TorchFactorGraph()
        z = fg.variable_from(scores / temperature)
        fg.add(Budget(z, budget=budget))
        fg.solve(max_iter=max_iter)
        return z.value

    def create_sent_mask(self, sent_lengths, max_sent_length):
        max_sent_length = max_sent_length if self.args.dataparallel else sent_lengths.max().item()
        return torch.arange(max_sent_length).expand(sent_lengths.size(0), max_sent_length).to(sent_lengths.device) < sent_lengths.unsqueeze(1)
    
    def convert_z_from_sentence_to_token_level(self, z, sent_starts, sent_ends, fix_end_positions, attention_mask):
        device = z.device
        mask = []
        sent_token_lengths = sent_ends - sent_starts
        for z_, sent_token_lengths_, fix_end_positions_ in zip(z, sent_token_lengths, fix_end_positions):
            fixed = torch.ones(1, fix_end_positions_).to(device)
            mask_ = torch.repeat_interleave(z_, sent_token_lengths_, dim=0).unsqueeze(0)
            mask_ = torch.cat((fixed, mask_), dim=1).squeeze(0)
            last_pos = mask_.size(0)
            pad_length = max(0, self.args.max_length - last_pos)
            mask_ = F.pad(mask_, pad=(0, pad_length), mode='constant', value=0)
            mask_[last_pos] = 1.0  # set the trailing [SEP] on
            mask.append(mask_)
        mask = torch.stack(mask)
        return mask

    def forward_eval(self, batch, intervene=None):
        with torch.no_grad():
            return self.forward(batch, mode='eval', intervene=intervene)


class VIBSentimentTokenModel(nn.Module):
    def __init__(self, args):
        super(VIBSentimentTokenModel, self).__init__()
        self.args = args
        self.encoder = TokenLevelEncoder(args)
        self.rep_to_logit_layer = nn.Sequential(nn.Linear(args.encoder_hidden_size, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 1))
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = Decoder(args)
    
    def forward(self, batch, mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        token_reps = self.encoder(input_ids, attention_mask)

        batch_size, num_tokens, _ = token_reps.size()
        token_logits = self.rep_to_logit_layer(token_reps)
        token_logits = self.drop(token_logits)
        token_logits = token_logits.squeeze(2)
        token_z = self.sample_z(token_logits, attention_mask, mode)

        output = self.decoder(input_ids, token_z, labels)

        kl_loss = self.calc_kl_loss(token_logits, attention_mask)
        loss = output.loss + self.args.beta * kl_loss

        return {
            'loss': loss,
            'pred_loss': output.loss,
            'logits': output.logits,
            'kl_loss': kl_loss,
            'token_z': token_z,
        }
    
    def sample_z(self, logits, mask, mode):
        device = logits.device
        batch_size = logits.size(0)
        lengths = mask.sum(dim=1)

        if mode == 'train':
            relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=logits)
            z = relaxed_bernoulli_dist.rsample()
        elif mode == 'eval':
            z = torch.sigmoid(logits)
        # mask out paddings
        z = z.masked_fill(~mask.bool(), 0.0)

        if mode == 'eval':
            active_top_k = (lengths * self.args.pi).ceil().long()

            # this is essentially sorting from large to small
            _, z_hard_inds = z.topk(z.size(-1), dim=-1)
            z_hard = torch.zeros(z.size()).long().to(device)

            # TODO: must be a better way than using a loop here
            for i in range(z.size(0)):
                subidx = z_hard_inds[i, :active_top_k[i]]
                z_hard[i, subidx] = 1.0
            z = z_hard
        return z

    def calc_kl_loss(self, logits, mask):
        p = Bernoulli(logits=logits)
        prior_probs = torch.ones(logits.size()).to(logits.device) * self.args.pi
        r = Bernoulli(probs=prior_probs)
        kl_loss = kl_divergence(p, r)
        kl_loss = kl_loss * mask.float()
        kl_loss = kl_loss.sum(dim=1) / (mask.sum(dim=1))
        return kl_loss.mean()

    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')


class FullContextSentimentModel(nn.Module):
    def __init__(self, args):
        super(FullContextSentimentModel, self).__init__()
        self.args = args
        self.decoder = Decoder(args)
    
    def forward(self, batch, mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self.decoder(input_ids, attention_mask, labels)

        loss = output.loss
        return {
            'loss': loss,
            'logits': output.logits,
        }

    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')


class SpectraSentimentTokenModel(nn.Module):
    def __init__(self, args):
        super(SpectraSentimentTokenModel, self).__init__()
        self.args = args
        self.encoder = TokenLevelEncoder(args)
        self.rep_to_logit_layer = nn.Sequential(nn.Linear(args.encoder_hidden_size, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 1))
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = Decoder(args)

    def forward(self, batch, mode='train', intervene=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        device = input_ids.device
        labels = batch['labels']

        token_reps = self.encoder(input_ids, attention_mask)
        batch_size, num_tokens, _ = token_reps.size()
        token_logits = self.rep_to_logit_layer(token_reps)
        token_logits = self.drop(token_logits)
        token_logits = token_logits.squeeze(2)
        token_logits = F.relu(token_logits)
        token_z = self.sample_z(token_logits, attention_mask, mode)
#        print(token_z.sum(dim=-1).unsqueeze(0) / attention_mask.sum(dim=-1).unsqueeze(0))
#        input()
        output = self.decoder(input_ids, token_z, labels)

        return {
            'logits': output.logits,
            'loss': output.loss,
            'token_z': token_z,
        }

    def sample_z(self, sent_logits, sent_mask, mode):
        device = sent_logits.device
        batch_size = sent_logits.size(0)
        sent_lengths = sent_mask.sum(dim=-1).long()
        z = []
        for i in range(batch_size):
            if self.args.budget_ratio is not None:
                budget = torch.round(self.args.budget_ratio * sent_lengths[i])
            elif self.args.budget is not None:
                budget = self.args.budget
            scores = sent_logits[i].unsqueeze(0)
            scores[0, ~sent_mask[i].bool()] = -1000.0
            z_probs = self.matching_smap_budget(
                scores,
                budget=budget,
                temperature=self.args.temperature,
                max_iter=self.args.solver_iter
            )
            z.append(z_probs)

        z = torch.cat(z, dim=0)
        return z

    def matching_smap_budget(self, scores, max_iter=100, temperature=1.0, init=True, budget=None):
        """
        M:Budget strategy for matchings extraction
        """
        m, n = scores.shape
        fg = TorchFactorGraph()
        z = fg.variable_from(scores / temperature)
        fg.add(Budget(z, budget=budget))
        fg.solve(max_iter=max_iter)
        return z.value

    def forward_eval(self, batch, intervene=None):
        with torch.no_grad():
            return self.forward(batch, mode='eval', intervene=intervene)
