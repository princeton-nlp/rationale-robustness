import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal

from rrtl.models.encoders import (
    TokenLevelEncoder,
    SentenceLevelEncoder,
)

from rrtl.models.decoders import (
    QADecoder,
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


import numpy as np
import math

gc = []
ac = []

class FullContextSQUADModel(nn.Module):
    def __init__(self, args):
        super(FullContextSQUADModel, self).__init__()
        self.args = args
        self.decoder = QADecoder(args)
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        if self.args.use_gold_rationale:
            self.rep_to_logit_layer = nn.Sequential(nn.Linear(args.encoder_hidden_size * 2, 128),
                                                    nn.ReLU(),
                                                    nn.Linear(128, 1))
    
    def forward(self, batch, mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        start_positions = labels[:, 0]
        end_positions = labels[:, 1]
        output = self.decoder(
            input_ids,
            attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        loss = output.loss

        if self.args.use_gold_rationale:
            sent_lengths = batch['sent_lengths']
            sent_starts = batch['sent_starts'][:, 1:]
            sent_ends = batch['sent_ends'][:, 1:]
            gold_sent_positions = batch['gold_sent_positions']
            sequence_output = output.hidden_states[-1]
            sequence_output = sequence_output * attention_mask.unsqueeze(-1).float() 
            sequence_output = self.drop(sequence_output)
            sentence_rep_shape = (sequence_output.shape[0], sent_starts.shape[1], sequence_output.shape[-1])    
            sentence_representations = torch.cat(
                ( 
                    sequence_output.gather(dim=1, index=sent_starts.unsqueeze(-1).expand(sentence_rep_shape)), 
                    sequence_output.gather(dim=1,index=sent_ends.unsqueeze(-1).expand(sentence_rep_shape))), 
                    dim=-1 
                )
            sent_mask = self.create_sent_mask(sent_lengths, batch['max_sent_length']).float()
            evidence_logits = self.rep_to_logit_layer(sentence_representations).squeeze(-1)
            sent_rationale_loss = self.calc_sent_rationale_loss(evidence_logits, gold_sent_positions, sent_mask)
            loss += self.args.gamma * sent_rationale_loss

#            self.calc_gc_ac(evidence_logits, gold_sent_positions, sent_mask)
            
            return {
                'loss': loss,
                'start_logits': output.start_logits,
                'end_logits': output.end_logits,
                'pred_loss': output.loss,
                'sent_z': sent_z,
            }
        else:
            return {
                'loss': loss,
                'start_logits': output.start_logits,
                'end_logits': output.end_logits,
                'pred_loss': output.loss,
            }

    def calc_gc_ac(self, evidence_logits, gold_sent_positions, sent_mask):
        last_pos = sent_mask.sum().long().item() - 1
#        print(gold_sent_positions)
        gold = set(gold_sent_positions.tolist())
        pred = np.array(evidence_logits[0].tolist())
#        print(pred)
        k = math.ceil(sent_mask.sum().item() * 0.7)
#        print(k)
        pred = set(pred.argsort()[-k:][::-1].tolist())
#        print(sent_mask.sum())
#        print(gold_sent_positions)
#        print(k)
        print(gold)
        print(pred)
#        input()
        gc.append(len(gold & pred) != 0)
#        ac.append(0 in pred)
        ac.append(last_pos in pred)
        print('gc:', sum(gc) / len(gc))
        print('ac:', sum(ac) / len(ac))
    
    def calc_sent_rationale_loss(self, evidence_logits, gold_sent_positions, sent_mask):
        device = evidence_logits.device
        rationale = torch.zeros(sent_mask.size()).to(device)
        rationale = rationale.scatter(1, gold_sent_positions.unsqueeze(1), 1.0)
        active_loss = sent_mask.view(-1) == 1
        active_logits = evidence_logits.view(-1)[active_loss]
        active_labels = rationale.float().view(-1)[active_loss]
        sent_rationale_loss = F.binary_cross_entropy_with_logits(active_logits, active_labels)
        return sent_rationale_loss

    def create_sent_mask(self, sent_lengths, max_sent_length):
        max_sent_length = max_sent_length if self.args.dataparallel else sent_lengths.max().item()
        return torch.arange(max_sent_length).expand(sent_lengths.size(0), max_sent_length).to(sent_lengths.device) < sent_lengths.unsqueeze(1)

    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')


class VIBSQUADSentModel(nn.Module):
    def __init__(self, args):
        super(VIBSQUADSentModel, self).__init__()
        self.args = args
        self.encoder = SentenceLevelEncoder(args)
        if self.args.mask_scoring_func == 'linear':
            self.rep_to_logit_layer = nn.Sequential(nn.Linear(args.encoder_hidden_size * 2, 128),
                                                    nn.ReLU(),
                                                    nn.Linear(128, 1))
        elif self.args.mask_scoring_func == 'query':
            self.question_rep_proj = nn.Linear(args.encoder_hidden_size * 2, 128)
            self.context_rep_proj = nn.Linear(args.encoder_hidden_size * 2, 128)
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = QADecoder(args)
    
    def forward(self, batch, mode='train', intervene=None):
#        input_ids = batch['input_ids']
#        attention_mask = batch['attention_mask']
#        device = input_ids.device
#        batch_size = input_ids.size(0)
#
#        labels = batch['labels']
#        start_positions = labels[:, 0]
#        end_positions = labels[:, 1]
#
#        question_input_ids = batch['question_input_ids']
#        question_attn_mask = batch['question_attn_mask']
#        context_input_ids = batch['context_input_ids']
#        context_attn_mask = batch['context_attn_mask']
#
#        sent_starts = batch['sent_starts']
#        sent_ends = batch['sent_ends']
#        sent_lengths = batch['sent_lengths']
#        question_end_positions = batch['question_end_positions']
#        gold_sent_positions = batch['gold_sent_positions']
#
#        question_rep, _ = self.encoder(
#            question_input_ids,
#            question_attn_mask,
#            sent_starts[:, 0:1],
#            sent_ends[:, 0:1]
#        )
#        context_reps, _ = self.encoder(
#            context_input_ids,
#            context_attn_mask,
#            sent_starts[:, 1:],
#            sent_ends[:, 1:]
#        )
#        question_rep = self.question_rep_proj(question_rep)
#        context_reps = self.context_rep_proj(context_reps)
#        sent_logits = F.cosine_similarity(context_reps, question_rep, dim=-1) / self.args.temperature

#        print(question_rep)
#        print(context_reps)
#        print(sent_logits)

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        start_positions = labels[:, 0]
        end_positions = labels[:, 1]

        device = input_ids.device

        sent_starts = batch['sent_starts']
        sent_ends = batch['sent_ends']
        sent_lengths = batch['sent_lengths']
        question_end_positions = batch['question_end_positions']
        gold_sent_positions = batch['gold_sent_positions']

        
        sent_reps, token_reps = self.encoder(input_ids, attention_mask, sent_starts, sent_ends)
        batch_size, num_tokens, _ = token_reps.size()

        # separate question and passage representations
        question_rep = sent_reps[:, 0:1]
        sent_reps = sent_reps[:, 1:]

        if self.args.mask_scoring_func == 'linear':
            sent_logits = self.rep_to_logit_layer(sent_reps)
            sent_logits = self.drop(sent_logits)
            sent_logits = sent_logits.squeeze(2)
        elif self.args.mask_scoring_func == 'query':
            sent_reps = self.rep_proj(sent_reps)
            question_rep = self.rep_proj(question_rep)
            sent_logits = (sent_reps * question_rep).sum(dim=-1)
        elif self.args.mask_scoring_func == 'random':
            sent_logits = torch.randn_like(sent_reps.sum(dim=2))

        sent_mask = self.create_sent_mask(sent_lengths, batch['max_sent_length']).float()

        sent_z = self.sample_z(
            sent_logits,
            sent_mask,
            question_end_positions,
            gold_sent_positions,
            mode
        )

        if intervene is not None:
            if 'add-gold' in intervene:
                sent_z[torch.arange(batch_size).to(device), gold_sent_positions] = 1.0
            if 'add-attack' in intervene:
                sent_z[:, -1] = 1.0
            if 'rm-gold' in intervene:
                sent_z[torch.arange(batch_size).to(device), gold_sent_positions] = 0.0
            if 'rm-attack' in intervene:
                sent_z[:, -1] = 0.0

        token_z = self.convert_z_from_sentence_to_token_level(
            sent_z,
            sent_starts[:, 1:],
            sent_ends[:, 1:],
            question_end_positions,
            attention_mask
        )
#        print(sent_z)
#        print(token_z)
        output = self.decoder(
            input_ids,
            token_z,
            start_positions=start_positions,
            end_positions=end_positions
        )
        loss = output.loss
        
        kl_loss = self.calc_kl_loss(sent_logits, sent_mask, gold_sent_positions)
        loss += self.args.beta * kl_loss
        if self.args.use_gold_rationale and mode == 'train':
            sent_rationale_loss = self.calc_sent_rationale_loss(sent_logits, gold_sent_positions, sent_mask)
            loss += self.args.gamma * sent_rationale_loss
        else:
            sent_rationale_loss = torch.zeros(loss.size()).to(loss.device)
        
        if self.args.use_neg_rationale and mode == 'train':
            neg_rationale_positions = batch['insert_positions']
            neg_rationale_loss = self.calc_neg_rationale_loss(sent_logits, neg_rationale_positions, sent_mask)
            loss += self.args.gamma2 * neg_rationale_loss
        else:
            neg_rationale_loss = torch.zeros(loss.size()).to(loss.device)
        
        gold_sent_probs = sent_z[torch.arange(sent_z.size(0)), gold_sent_positions]

        return {
            'loss': loss,
            'pred_loss': output.loss,
            'kl_loss': kl_loss,
            'start_logits': output.start_logits,
            'end_logits': output.end_logits,
            'token_z': token_z,
            'sent_z': sent_z,
            'sent_rationale_loss': sent_rationale_loss,
            'neg_rationale_loss': neg_rationale_loss,
            'gold_sent_probs': gold_sent_probs,
        }

    def sample_z(self, sent_logits, sent_mask, question_end_positions, gold_sent_positions, mode):
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
#        print(z)

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
    
    def create_sent_mask(self, sent_lengths, max_sent_length):
        max_sent_length = max_sent_length if self.args.dataparallel else sent_lengths.max().item()
        return torch.arange(max_sent_length).expand(sent_lengths.size(0), max_sent_length).to(sent_lengths.device) < sent_lengths.unsqueeze(1)
    
    def convert_z_from_sentence_to_token_level(self, z, sent_starts, sent_ends, question_end_positions, attention_mask):
        device = attention_mask.device
        mask = []
        sent_token_lengths = sent_ends - sent_starts
        for z_, sent_token_lengths_, question_end_positions_ in zip(z, sent_token_lengths, question_end_positions):
            fixed = torch.ones(1, question_end_positions_).to(device)
            mask_ = torch.repeat_interleave(z_, sent_token_lengths_, dim=0).unsqueeze(0)
            mask_ = torch.cat((fixed, mask_), dim=1).squeeze(0)
            last_pos = mask_.size(0)
            pad_length = max(0, self.args.max_length - last_pos)
            mask_ =  F.pad(mask_, pad=(0, pad_length), mode='constant', value=0)
            mask_[last_pos] = 1.0  # set the trailing [SEP] on
            mask.append(mask_)
        mask = torch.stack(mask)
        return mask

    def calc_kl_loss(self, logits, mask, gold_sent_positions):
        p = Bernoulli(logits=logits)

        prior_probs = torch.ones(logits.size()).to(logits.device) * self.args.pi
        if self.args.flexible_prior == 'small-gold-fixed':
            prior_probs[torch.arange(logits.size(0)).to(logits.device), gold_sent_positions] += 0.1
        elif self.args.flexible_prior == 'mid-gold-fixed':
            prior_probs[torch.arange(logits.size(0)).to(logits.device), gold_sent_positions] += 0.15
        elif self.args.flexible_prior == 'large-gold-fixed':
            prior_probs[torch.arange(logits.size(0)).to(logits.device), gold_sent_positions] += 0.2
        elif self.args.flexible_prior == 'gold-fixed':
            prior_probs[torch.arange(logits.size(0)).to(logits.device), gold_sent_positions] = 0.99

        r = Bernoulli(probs=prior_probs)

        kl_loss = kl_divergence(p, r)
        kl_loss = kl_loss * mask.float()
        kl_loss = kl_loss.sum(dim=1) / mask.sum(dim=1)
        return kl_loss.mean()
    
    def calc_sent_rationale_loss(self, sent_logits, gold_sent_positions, sent_mask):
        device = sent_logits.device

        rationale = torch.zeros(sent_mask.size()).to(device)
        rationale += self.args.flexible_gold[0] 
        rationale = rationale.scatter(1, gold_sent_positions.unsqueeze(1), self.args.flexible_gold[1])
        #rationale = torch.zeros(sent_mask.size()).to(device).scatter(1, gold_sent_positions.unsqueeze(1), 1.0)
        active_loss_pos = (sent_mask.view(-1) == 1)
        active_logits = sent_logits.view(-1)[active_loss_pos]
        active_labels = rationale.float().view(-1)[active_loss_pos]
        sent_rationale_loss = F.binary_cross_entropy_with_logits(active_logits, active_labels)
        return sent_rationale_loss

    def calc_neg_rationale_loss(self, sent_logits, neg_rationale_positions, sent_mask):
        device = sent_logits.device
        rationale = torch.zeros(sent_mask.size()).to(device).scatter(1, neg_rationale_positions, 0.0)
        active_loss_pos = (sent_mask.view(-1) == 1)
        active_logits = sent_logits.clone().view(-1)[active_loss_pos]
        active_labels = rationale.float().view(-1)[active_loss_pos]
        neg_rationale_loss = F.binary_cross_entropy_with_logits(active_logits, active_labels)
        return neg_rationale_loss

    def forward_eval(self, batch, intervene=None):
        with torch.no_grad():
            return self.forward(batch, mode='eval', intervene=intervene)


class SpectraSQUADModel(nn.Module):
    def __init__(self, args):
        super(SpectraSQUADModel, self).__init__()
        self.args = args
        self.encoder = SentenceLevelEncoder(args)
        self.question_rep_proj = nn.Linear(args.encoder_hidden_size * 2, 128)
        self.context_rep_proj = nn.Linear(args.encoder_hidden_size * 2, 128)
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = QADecoder(args)

    def forward(self, batch, mode='train', intervene=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        device = input_ids.device
        batch_size = input_ids.size(0)

        labels = batch['labels']
        start_positions = labels[:, 0]
        end_positions = labels[:, 1]

        question_input_ids = batch['question_input_ids']
        question_attn_mask = batch['question_attn_mask']
        context_input_ids = batch['context_input_ids']
        context_attn_mask = batch['context_attn_mask']

        sent_starts = batch['sent_starts']
        sent_ends = batch['sent_ends']
        sent_lengths = batch['sent_lengths']
        question_end_positions = batch['question_end_positions']
        gold_sent_positions = batch['gold_sent_positions']

        question_rep, _ = self.encoder(
            question_input_ids,
            question_attn_mask,
            sent_starts[:, 0:1],
            sent_ends[:, 0:1]
        )
        context_reps, _ = self.encoder(
            context_input_ids,
            context_attn_mask,
            sent_starts[:, 1:],
            sent_ends[:, 1:]
        )
        question_rep = self.question_rep_proj(question_rep)
        context_reps = self.context_rep_proj(context_reps)
        sent_logits = (context_reps * question_rep).sum(dim=-1)
        #sent_logits = F.cosine_similarity(context_reps, question_rep, dim=-1)

        sent_mask = self.create_sent_mask(sent_lengths, batch['max_sent_length']).float()
        sent_z = self.sample_z(
            sent_logits,
            sent_mask,
            question_end_positions,
            gold_sent_positions,
            mode
        )
        token_z = self.convert_z_from_sentence_to_token_level(
            sent_z,
            sent_starts[:, 1:],
            sent_ends[:, 1:],
            question_end_positions,
            attention_mask
        )
        output = self.decoder(
            input_ids,
            token_z,
            start_positions=start_positions,
            end_positions=end_positions
        )
        loss = output.loss
        
        gold_sent_probs = sent_z[torch.arange(sent_z.size(0)), gold_sent_positions]

        if self.args.use_gold_rationale and mode == 'train':
            sent_rationale_loss = self.calc_sent_rationale_loss(
                sent_logits,
                gold_sent_positions,
                sent_mask
            )
            loss += self.args.gamma * sent_rationale_loss
        else:
            sent_rationale_loss = torch.zeros(loss.size()).to(loss.device)

        return {
            'loss': loss,
            'pred_loss': output.loss,
            'start_logits': output.start_logits,
            'end_logits': output.end_logits,
            'token_z': token_z,
            'sent_z': sent_z,
            'gold_sent_probs': gold_sent_probs,
            'sent_rationale_loss': sent_rationale_loss,
        }

    def sample_z(self, sent_logits, sent_mask, question_end_positions, gold_sent_positions, mode):
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
#            scores = torch.rand(scores.size(), requires_grad=True).to(device)
            z_probs = self.matching_smap_budget(
                scores,
                budget=budget,
                temperature=self.args.temperature,
                max_iter=self.args.solver_iter
            )
#            print('scores:', scores[0])
#            print('z:', z_probs[0])
#            input()
            z.append(z_probs)

        z = torch.cat(z, dim=0)
#        z.register_hook(lambda grad: print('z', grad))
#        sent_logits.register_hook(lambda grad: print('s', grad))
#        print('---')
#        print('z:', z)
#        print('sent_logits:', sent_logits)
#        input()
        return z

    def matching_smap_budget(self, scores, max_iter=100, temperature=1.0, init=True, budget=None):
        """
        M:Budget strategy for matchings extraction
        """
        m, n = scores.shape
        fg = TorchFactorGraph()
        z = fg.variable_from(scores / temperature)
#        for i in range(m):
#            fg.add(Or(z[i, :]))
#        for j in range(n):
#            fg.add(Or(z[:, j]))
#        for i in range(m):
#            fg.add(AtMostOne(z[i, :]))
#        for j in range(n):
#            fg.add(AtMostOne(z[:, j]))  # some cols may be 0
        fg.add(Budget(z, budget=budget))
        fg.solve(max_iter=max_iter)
        return z.value

    def create_sent_mask(self, sent_lengths, max_sent_length):
        max_sent_length = max_sent_length if self.args.dataparallel else sent_lengths.max().item()
        return torch.arange(max_sent_length).expand(sent_lengths.size(0), max_sent_length).to(sent_lengths.device) < sent_lengths.unsqueeze(1)
    
    def convert_z_from_sentence_to_token_level(self, z, sent_starts, sent_ends, question_end_positions, attention_mask):
        device = z.device
        mask = []
        sent_token_lengths = sent_ends - sent_starts
        for z_, sent_token_lengths_, question_end_positions_ in zip(z, sent_token_lengths, question_end_positions):
            fixed = torch.ones(1, question_end_positions_).to(device)
            mask_ = torch.repeat_interleave(z_, sent_token_lengths_, dim=0).unsqueeze(0)
            mask_ = torch.cat((fixed, mask_), dim=1).squeeze(0)
            last_pos = mask_.size(0)
            pad_length = max(0, self.args.max_length - last_pos)
            mask_ = F.pad(mask_, pad=(0, pad_length), mode='constant', value=0)
            mask_[last_pos] = 1.0  # set the trailing [SEP] on
            mask.append(mask_)
        mask = torch.stack(mask)
        return mask

    def calc_sent_rationale_loss(self, sent_logits, gold_sent_positions, sent_mask):
        device = sent_logits.device

        rationale = torch.zeros(sent_mask.size()).to(device)
        rationale += self.args.flexible_gold[0] 
        rationale = rationale.scatter(1, gold_sent_positions.unsqueeze(1), self.args.flexible_gold[1])
        #rationale = torch.zeros(sent_mask.size()).to(device).scatter(1, gold_sent_positions.unsqueeze(1), 1.0)
        active_loss_pos = (sent_mask.view(-1) == 1)
        active_logits = sent_logits.view(-1)[active_loss_pos]
        active_labels = rationale.float().view(-1)[active_loss_pos]
        sent_rationale_loss = F.binary_cross_entropy_with_logits(active_logits, active_labels)
        return sent_rationale_loss

    def forward_eval(self, batch, intervene=None):
        with torch.no_grad():
            return self.forward(batch, mode='eval', intervene=intervene)


class VIBSQUADTokenModel(nn.Module):
    def __init__(self, args):
        super(VIBSQUADTokenModel, self).__init__()
        self.args = args
        self.encoder = TokenLevelEncoder(args)
        self.rep_to_logit_layer = nn.Sequential(nn.Linear(args.encoder_hidden_size, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 1))
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = QADecoder(args)
    
    def forward(self, batch, mode='train', intervene=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        start_positions = labels[:, 0]
        end_positions = labels[:, 1]

        device = input_ids.device

        sent_starts = batch['sent_starts']
        sent_ends = batch['sent_ends']
        sent_lengths = batch['sent_lengths']
        question_end_positions = batch['question_end_positions']
        gold_sent_positions = batch['gold_sent_positions']

        reps = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.rep_to_logit_layer(reps)
        logits = self.drop(logits)
        logits = logits.squeeze(2)

        token_z = self.sample_z(logits, attention_mask, mode, question_end_positions)

        output = self.decoder(input_ids, token_z, start_positions, end_positions)

        kl_loss = self.calc_kl_loss(logits, attention_mask)
        loss = output.loss + self.args.beta * kl_loss
        return {
            'loss': loss,
            'pred_loss': output.loss,
            'kl_loss': kl_loss,
            'start_logits': output.start_logits,
            'end_logits': output.end_logits,
            'token_z': token_z,
        }

    def sample_z(self, logits, mask, mode, question_end_positions):
        device = logits.device
        batch_size = logits.size(0)
        lengths = mask.sum(dim=1)

        question_mask = (torch.arange(logits.size(1)).expand(batch_size, logits.size(1)).to(device) < question_end_positions.unsqueeze(1)).long()

        if mode == 'train':
            relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=logits)
            z = relaxed_bernoulli_dist.rsample()  # sample with reparameterization
            z = z.masked_fill(question_mask.bool(), 1.0)
        elif mode == 'eval':
            z = torch.sigmoid(logits)
#            print(z)
        # mask out paddings
        z = z.masked_fill(~mask.bool(), 0.0)

        if mode == 'eval':
            z = z.masked_fill(question_mask.bool(), 1.0)
            active_top_k = ((lengths - question_mask.sum(-1)) * self.args.pi).ceil().long()

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
        kl_loss = kl_loss.sum(dim=1) / mask.sum(dim=1)
        return kl_loss.mean()

    def forward_eval(self, batch, intervene=None):
        with torch.no_grad():
            return self.forward(batch, mode='eval', intervene=intervene)


class SpectraSQUADTokenModel(nn.Module):
    def __init__(self, args):
        super(SpectraSQUADTokenModel, self).__init__()
        self.args = args
        self.encoder = TokenLevelEncoder(args)
        self.question_rep_proj = nn.Linear(args.encoder_hidden_size, 128)
        self.context_rep_proj = nn.Linear(args.encoder_hidden_size, 128)
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = QADecoder(args)

    def forward(self, batch, mode='train', intervene=None):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        device = input_ids.device
        batch_size = input_ids.size(0)

        labels = batch['labels']
        start_positions = labels[:, 0]
        end_positions = labels[:, 1]
        question_end_positions = batch['question_end_positions']

        question_input_ids = batch['question_input_ids']
        question_attn_mask = batch['question_attn_mask']
        context_input_ids = batch['context_input_ids']
        context_attn_mask = batch['context_attn_mask']

        question_reps = self.encoder(question_input_ids, question_attn_mask)
        question_reps = question_reps[:, 0, :].unsqueeze(1)
        context_reps = self.encoder(context_input_ids, context_attn_mask)
        question_reps = self.question_rep_proj(question_reps)
        context_reps = self.context_rep_proj(context_reps)
        match_scores = torch.einsum('bid,bjd->bij', question_reps, context_reps)
        match_mask = context_attn_mask.unsqueeze(1).float()
#        match_mask = torch.einsum('bi,bj->bij', question_attn_mask.float(), context_attn_mask.float())

        token_z = self.sample_z(
            match_scores,
            match_mask,
            question_end_positions,
            mode
        )
        output = self.decoder(
            input_ids,
            token_z,
            start_positions=start_positions,
            end_positions=end_positions
        )
        loss = output.loss

        return {
            'loss': loss,
            'pred_loss': output.loss,
            'start_logits': output.start_logits,
            'end_logits': output.end_logits,
            'token_z': token_z,
        }

    def sample_z(self, match_scores, match_mask, question_end_positions, mode):
        device = match_scores.device
        batch_size = match_scores.size(0)
        z = []
        for i in range(batch_size):
            if self.args.budget_ratio is not None:
                budget = torch.round(self.args.budget_ratio * sent_lengths[i])
            elif self.args.budget is not None:
                budget = self.args.budget
            scores = match_scores[i]  # q x c
            m = (match_mask[i].long() - 1) * 100000.0
#            print(match_mask[i].sum())

#            print(question_end_positions[i])
            scores = scores + m
            z_probs = self.matching_smap_budget(
                scores,
                budget=budget,
                temperature=self.args.temperature,
                max_iter=self.args.solver_iter
            )
            
            # TODO: double check correctness #########
            context_length = match_mask[i].sum().long() - 1
            z_probs_concat = torch.zeros(self.args.max_length).to(device)
            z_probs_concat[0:question_end_positions[i]] = 1.0
            z_probs_concat[question_end_positions[i]:question_end_positions[i] + context_length] = z_probs[0, 1:context_length + 1]
            z.append(z_probs_concat.unsqueeze(0))
            ##########################################
        z = torch.cat(z, dim=0)
#        print(z)
#        print(z.sum())
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

    def matching_smap_xor_budget(self, scores, max_iter=100, temperature=1.0, init=True, budget=None):
        """
        M:Budget strategy for matchings extraction
        """
        m, n = scores.shape
        fg = TorchFactorGraph()
        z = fg.variable_from(scores / temperature)
        fg.add(Budget(z, budget=budget))
        for i in range(m):
            fg.add(Xor(z[i, :]))
        for j in range(n):
            fg.add(AtMostOne(z[:, j]))  # some cols may be 0
        fg.solve(max_iter=max_iter)
        return z.value.cuda()

    def forward_eval(self, batch, intervene=None):
        with torch.no_grad():
            return self.forward(batch, mode='eval', intervene=intervene)