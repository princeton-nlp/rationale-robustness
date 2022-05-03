import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence

from transformers import BertModel

from rrtl.transformer_models.modeling_bert import (
    #  BertModel,
    BertForSequenceClassification,
    BertForQuestionAnswering,
)

from rrtl.transformer_models.modeling_distilbert import (
    DistilBertModel,
    DistilBertForSequenceClassification,
    DistilBertForQuestionAnswering,
)


def get_encoder_model_class(args):
    if args.encoder_type == 'bert-base-uncased':
        return BertModel
    elif args.encoder_type == 'distilbert-base-uncased':
        return DistilBertModel


def get_decoder_model_class(args):
    if args.decoder_type == 'bert-base-uncased':
        return {
            'classification': BertForSequenceClassification, 
            'qa': BertForQuestionAnswering,
        }
    elif args.decoder_type == 'distilbert-base-uncased':
        return {
            'classification': DistilBertForSequenceClassification, 
            'qa': DistilBertForQuestionAnswering,
        }


class ClassificationDecoder(nn.Module):
    def __init__(self, args):
        super(ClassificationDecoder, self).__init__()
        self.args = args

        encoder_model_class = get_encoder_model_class(args)
        self.model = encoder_model_class.from_pretrained(
            args.decoder_type,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_dir=args.cache_dir,
        )
    
    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output


class QADecoder(nn.Module):
    def __init__(self, args):
        super(QADecoder, self).__init__()
        self.args = args

        decoder_model_classes = get_decoder_model_class(args)
        self.model = decoder_model_classes['qa'].from_pretrained(
            args.decoder_type,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_dir=self.args.cache_dir,
        )
    
    def forward(self, input_ids, attention_mask, start_positions, end_positions):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        return output


class FullContextNLIModel(nn.Module):
    def __init__(self, args):
        super(FullContextNLIModel, self).__init__()
        self.args = args
        self.decoder = ClassificationDecoder(args)
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
        # TODO: handle fix premise/hypothesis cases 
        # when --fix-input premise and --use-gold-rationale are both on
        active_loss_pos = (mask.view(-1) == 1)
        active_logits = logits.view(-1)[active_loss_pos]
        active_labels = rationale.float().view(-1)[active_loss_pos]
        loss = F.binary_cross_entropy_with_logits(active_logits, active_labels)
        return loss


class VIBNLIModel(nn.Module):
    def __init__(self, args):
        super(VIBNLIModel, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained(args.encoder_type, cache_dir=self.args.cache_dir)
        self.rep_to_logit_layer = nn.Linear(args.encoder_hidden_size, 1)
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = ClassificationDecoder(args)

    def forward(self, batch, mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        special_token_pos = batch['special_token_pos']
        premise_pos = batch['premise_pos']
        hypothesis_pos = batch['hypothesis_pos']

        reps = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.rep_to_logit_layer(reps)
        logits = self.drop(logits)
        logits = logits.squeeze(2)
        
        z = self.sample_z(logits, attention_mask, mode, special_token_pos, premise_pos, hypothesis_pos)
        output = self.decoder(input_ids, z, labels)

        kl_loss = self.calc_kl_loss(logits, attention_mask)
        loss = output.loss + self.args.beta * kl_loss

        if self.args.use_gold_rationale:
            rationales = batch['rationales']
            gold_rationale_loss = self.calc_rationale_loss(logits, rationales, attention_mask)
            loss += self.args.gamma * gold_rationale_loss
        
        if self.args.use_neg_rationale:
            neg_rationales = batch['neg_rationales']
            neg_rationale_loss = self.calc_rationale_loss(logits, 1 - neg_rationales, attention_mask)
            loss += self.args.gamma2 * neg_rationale_loss

        return {
            'loss': loss,
            'logits': output.logits,
            'rationales': z,
            'kl_loss': kl_loss,
        }
    
    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')
    
    def sample_z(self, logits, mask, mode, special_token_pos, premise_pos, hypothesis_pos):
        device = logits.device
        batch_size = logits.size(0)
        lengths = mask.sum(dim=1)

        if mode == 'train':
            relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=logits)
            z = relaxed_bernoulli_dist.rsample()  # sample with reparameterization
        elif mode == 'eval':
            # TODO: check if this is the right inference logic,
            # do we need to sample here using the gumbel-softmax?
            z = torch.sigmoid(logits)

        # mask out paddings
        z = z.masked_fill(~mask.bool(), 0.0)

        # TODO: refactor fix input (p or h) logic to make it cleaner
        if self.args.fix_input is not None:
            if self.args.fix_input == 'premise':
                starts = premise_pos[:, 0]
                ends = premise_pos[:, 1] + 1
            elif self.args.fix_input == 'hypothesis':
                starts = hypothesis_pos[:, 0]
                ends = hypothesis_pos[:, 1] + 1
            start_mask = (torch.arange(logits.size(1)).expand(batch_size, logits.size(1)).to(device) < starts.unsqueeze(1)).long()
            end_mask = (torch.arange(logits.size(1)).expand(batch_size, logits.size(1)).to(device) < ends.unsqueeze(1)).long()
            input_mask = (end_mask - start_mask).bool()
            z = z.masked_fill(input_mask, 0.0)

        if mode == 'eval':
            num_input_mask_tokens = input_mask.sum(dim=1) if self.args.fix_input in ('premise', 'hypothesis') else 0
            num_special_tokens = 3
            z = z.scatter(1, special_token_pos, 0.0)
            active_top_k = ((lengths - num_special_tokens - num_input_mask_tokens) * self.args.pi).ceil().long()
            # this is essentially sorting from large to small
            _, z_hard_inds = z.topk(z.size(-1), dim=-1)
            z_hard = torch.zeros(z.size()).long().to(device)

            # TODO: must be a better way than using a loop here
            for i in range(z.size(0)):
                subidx = z_hard_inds[i, :active_top_k[i]]
                z_hard[i, subidx] = 1.0
            z = z_hard

        if self.args.fix_input is not None:
            z = z.masked_fill(input_mask, 1.0)  # TODO: refactor fix input (p or h) logic to make it cleaner
        z = z.scatter(1, special_token_pos, 1.0)  # always turn special tokens on
        return z

    def calc_kl_loss(self, logits, mask):
        p = Bernoulli(logits=logits)

        prior_probs = torch.ones(logits.size()).to(logits.device) * self.args.pi
        r = Bernoulli(probs=prior_probs)

        kl_loss = kl_divergence(p, r)
        kl_loss = kl_loss * mask.float()
        kl_loss = kl_loss.sum(dim=1) / mask.sum(dim=1)
        return kl_loss.mean()
    
    def calc_rationale_loss(self, logits, rationale, mask):
        # TODO: handle fix premise/hypothesis cases 
        # when --fix-input premise and --use-gold-rationale are both on
        active_loss_pos = (mask.view(-1) == 1)
        active_logits = logits.view(-1)[active_loss_pos]
        active_labels = rationale.float().view(-1)[active_loss_pos]
        loss = F.binary_cross_entropy_with_logits(active_logits, active_labels)
        return loss


class FullContextModel(nn.Module):
    def __init__(self, args):
        super(FullContextModel, self).__init__()
        self.args = args
        self.decoder = Decoder(args)
        if self.args.use_gold_rationale:
            self.linear = nn.Linear(self.args.encoder_hidden_size, 1)
            self.drop = nn.Dropout(self.args.dropout_rate)
    
    def forward(self, batch, mode='train'):
        input_ids = batch['orig_input_ids']
        attention_mask = batch['orig_attention_mask']
        labels = batch['orig_labels']
        output = self.decoder(input_ids, attention_mask, labels)
        loss = output.loss

        if self.args.use_gold_rationale:
            rationales = batch['orig_rationales']
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

class VIBModel(nn.Module):
    def __init__(self, args):
        super(VIBModel, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained(args.encoder_type, cache_dir=self.args.cache_dir)
        self.rep_to_logit_layer = nn.Linear(args.encoder_hidden_size, 1)
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = Decoder(args)

    def forward(self, batch, mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        special_token_pos = batch['special_token_pos']
        premise_pos = batch['premise_pos']
        hypothesis_pos = batch['hypothesis_pos']

        reps = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.rep_to_logit_layer(reps)
        logits = self.drop(logits)
        logits = logits.squeeze(2)
        
        z = self.sample_z(logits, attention_mask, mode, special_token_pos, premise_pos, hypothesis_pos)
        output = self.decoder(input_ids, z, labels)

        kl_loss = self.calc_kl_loss(logits, attention_mask)
        loss = output.loss + self.args.beta * kl_loss

        if self.args.use_gold_rationale:
            rationales = batch['rationales']
            gold_rationale_loss = self.calc_rationale_loss(logits, rationales, attention_mask)
            loss += self.args.gamma * gold_rationale_loss
        
        if self.args.use_neg_rationale:
            neg_rationales = batch['neg_rationales']
            neg_rationale_loss = self.calc_rationale_loss(logits, 1 - neg_rationales, attention_mask)
            loss += self.args.gamma2 * neg_rationale_loss

        return {
            'loss': loss,
            'logits': output.logits,
            'rationales': z,
            'kl_loss': kl_loss,
        }
    
    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')
    
    def sample_z(self, logits, mask, mode, special_token_pos, premise_pos, hypothesis_pos):
        device = logits.device
        batch_size = logits.size(0)
        lengths = mask.sum(dim=1)

        if mode == 'train':
            relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=logits)
            z = relaxed_bernoulli_dist.rsample()  # sample with reparameterization
        elif mode == 'eval':
            # TODO: check if this is the right inference logic,
            # do we need to sample here using the gumbel-softmax?
            z = torch.sigmoid(logits)

        # mask out paddings
        z = z.masked_fill(~mask.bool(), 0.0)

        # TODO: refactor fix input (p or h) logic to make it cleaner
        if self.args.fix_input is not None:
            if self.args.fix_input == 'premise':
                starts = premise_pos[:, 0]
                ends = premise_pos[:, 1] + 1
            elif self.args.fix_input == 'hypothesis':
                starts = hypothesis_pos[:, 0]
                ends = hypothesis_pos[:, 1] + 1
            start_mask = (torch.arange(logits.size(1)).expand(batch_size, logits.size(1)).to(device) < starts.unsqueeze(1)).long()
            end_mask = (torch.arange(logits.size(1)).expand(batch_size, logits.size(1)).to(device) < ends.unsqueeze(1)).long()
            input_mask = (end_mask - start_mask).bool()
            z = z.masked_fill(input_mask, 0.0)

        if mode == 'eval':
            num_input_mask_tokens = input_mask.sum(dim=1) if self.args.fix_input in ('premise', 'hypothesis') else 0
            num_special_tokens = 3
            z = z.scatter(1, special_token_pos, 0.0)
            active_top_k = ((lengths - num_special_tokens - num_input_mask_tokens) * self.args.pi).ceil().long()
            # this is essentially sorting from large to small
            _, z_hard_inds = z.topk(z.size(-1), dim=-1)
            z_hard = torch.zeros(z.size()).long().to(device)

            # TODO: must be a better way than using a loop here
            for i in range(z.size(0)):
                subidx = z_hard_inds[i, :active_top_k[i]]
                z_hard[i, subidx] = 1.0
            z = z_hard

        if self.args.fix_input is not None:
            z = z.masked_fill(input_mask, 1.0)  # TODO: refactor fix input (p or h) logic to make it cleaner
        z = z.scatter(1, special_token_pos, 1.0)  # always turn special tokens on
        return z

    def calc_kl_loss(self, logits, mask):
        p = Bernoulli(logits=logits)

        prior_probs = torch.ones(logits.size()).to(logits.device) * self.args.pi
        r = Bernoulli(probs=prior_probs)

        kl_loss = kl_divergence(p, r)
        kl_loss = kl_loss * mask.float()
        kl_loss = kl_loss.sum(dim=1) / mask.sum(dim=1)
        return kl_loss.mean()
    
    def calc_rationale_loss(self, logits, rationale, mask):
        # TODO: handle fix premise/hypothesis cases 
        # when --fix-input premise and --use-gold-rationale are both on
        active_loss_pos = (mask.view(-1) == 1)
        active_logits = logits.view(-1)[active_loss_pos]
        active_labels = rationale.float().view(-1)[active_loss_pos]
        loss = F.binary_cross_entropy_with_logits(active_logits, active_labels)
        return loss


class VIBSentimentModel(nn.Module):
    def __init__(self, args):
        super(VIBSentimentModel, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained(args.encoder_type, cache_dir=self.args.cache_dir)
        self.rep_to_logit_layer = nn.Linear(args.encoder_hidden_size, 1)
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = Decoder(args)

    # JH prepend orig
    def forward(self, batch, mode='train'):
        input_ids = batch['orig_input_ids']
        attention_mask = batch['orig_attention_mask']
        labels = batch['orig_labels']
        special_token_pos = batch['orig_special_token_pos']

        reps = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.rep_to_logit_layer(reps)
        logits = self.drop(logits)
        logits = logits.squeeze(2)
        
        z = self.sample_z(logits, attention_mask, mode, special_token_pos)
        output = self.decoder(input_ids, z, labels)

        kl_loss = self.calc_kl_loss(logits, attention_mask)
        loss = output.loss + self.args.beta * kl_loss

        return {
            'loss': loss,
            'logits': output.logits,
            'rationales': z,
            'kl_loss': kl_loss,
        }
    
    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')
    
    def sample_z(self, logits, mask, mode, special_token_pos):
        device = logits.device
        batch_size = logits.size(0)
        lengths = mask.sum(dim=1)

        if mode == 'train':
            relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=logits)
            z = relaxed_bernoulli_dist.rsample()  # sample with reparameterization
        elif mode == 'eval':
            # TODO: check if this is the right inference logic,
            # do we need to sample here using the gumbel-softmax?
            z = torch.sigmoid(logits)

        # mask out paddings
        z = z.masked_fill(~mask.bool(), 0.0)

        if mode == 'eval':
            num_input_mask_tokens = 0
            num_special_tokens = 2
            z = z.scatter(1, special_token_pos, 0.0)
            active_top_k = ((lengths - num_special_tokens - num_input_mask_tokens) * self.args.pi).ceil().long()
            # this is essentially sorting from large to small
            _, z_hard_inds = z.topk(z.size(-1), dim=-1)
            z_hard = torch.zeros(z.size()).long().to(device)

            # TODO: must be a better way than using a loop here
            for i in range(z.size(0)):
                subidx = z_hard_inds[i, :active_top_k[i]]
                z_hard[i, subidx] = 1.0
            z = z_hard

        z = z.scatter(1, special_token_pos, 1.0)  # always turn special tokens on
        return z

    def calc_kl_loss(self, logits, mask):
        p = Bernoulli(logits=logits)

        prior_probs = torch.ones(logits.size()).to(logits.device) * self.args.pi
        r = Bernoulli(probs=prior_probs)

        kl_loss = kl_divergence(p, r)
        kl_loss = kl_loss * mask.float()
        kl_loss = kl_loss.sum(dim=1) / mask.sum(dim=1)
        return kl_loss.mean()
    
    def calc_rationale_loss(self, logits, rationale, mask):
        # TODO: handle fix premise/hypothesis cases 
        # when --fix-input premise and --use-gold-rationale are both on
        active_loss_pos = (mask.view(-1) == 1)
        active_logits = logits.view(-1)[active_loss_pos]
        active_labels = rationale.float().view(-1)[active_loss_pos]
        loss = F.binary_cross_entropy_with_logits(active_logits, active_labels)
        return loss

class FullContextSQUADModel(nn.Module):
    def __init__(self, args):
        super(FullContextSQUADModel, self).__init__()
        self.args = args
        self.decoder = QADecoder(args)
    
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

        return {
            'loss': output.loss,
            'start_logits': output.start_logits,
            'end_logits': output.end_logits,
            'pred_loss': output.loss,
            'kl_loss': torch.tensor(0.0).to(input_ids.device),
        }

    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')


class VIBSQUADSentModel(nn.Module):
    def __init__(self, args):
        super(VIBSQUADSentModel, self).__init__()
        self.args = args
        self.encoder = SentenceLevelEncoder(args)
        self.rep_to_logit_layer = nn.Linear(args.encoder_hidden_size * 2, 1)
        self.drop = nn.Dropout(p=self.args.dropout_rate)
        self.decoder = QADecoder(args)
    
    def forward(self, batch, mode='train'):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        start_positions = labels[:, 0]
        end_positions = labels[:, 1]

        sent_starts = batch['sent_starts']
        sent_ends = batch['sent_ends']
        sent_lengths = batch['sent_lengths']
        fix_positions = batch['fix_positions']
        gold_sent_positions = batch['gold_sent_positions']

        sent_reps, token_reps = self.encoder(input_ids, attention_mask, sent_starts, sent_ends)
        batch_size, num_tokens, _ = token_reps.size()
        sent_logits = self.rep_to_logit_layer(sent_reps)
        sent_logits = self.drop(sent_logits)
        sent_logits = sent_logits.squeeze(2)

        sent_mask = self.create_sent_mask(sent_lengths, batch['max_sent_length']).float()

        sent_z = self.sample_z(sent_logits, sent_mask, fix_positions, gold_sent_positions, mode)
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
#            attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        loss = output.loss

        
        kl_loss = self.calc_kl_loss(sent_logits, sent_mask)
        loss += kl_loss

        if self.args.use_gold_rationale:
            gold_sent_rationale_loss = self.calc_sent_rationale_loss(sent_logits, gold_sent_positions, sent_mask)
            loss += self.args.gamma * gold_sent_rationale_loss
        else:
            gold_sent_rationale_loss = 0

        return {
            'loss': loss,
            'pred_loss': output.loss,
            'kl_loss': kl_loss,
            'start_logits': output.start_logits,
            'end_logits': output.end_logits,
            'token_z': token_z,
            'sent_z': sent_z,
            'gold_sent_rationale_loss': gold_sent_rationale_loss,
        }

    def sample_z(self, sent_logits, sent_mask, fix_positions, gold_sent_positions, mode):
        # TODO:
        # add [CLS], question, and [SEP] as sentence reps in dataloader
        # which need to be always turned on
        # also need to add currect sentence position to always make sure the
        # correct span sentence is not masked out
        device = sent_logits.device
        batch_size = sent_logits.size(0)
        lengths = sent_mask.sum(dim=1)

        if mode == 'train':
            relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=sent_logits)
            z = relaxed_bernoulli_dist.rsample()  # sample with reparameterization
        elif mode == 'eval':
            # TODO: check if this is the right inference logic,
            # do we need to sample here using the gumbel-softmax?
            z = torch.sigmoid(sent_logits)

        # mask out sentence-level paddings
        z = z.masked_fill(~sent_mask.bool(), 0.0)

        if mode == 'train':
            if not self.args.use_gold_rationale:
                z = z.scatter(1, gold_sent_positions.unsqueeze(1), 1.0)
        elif mode == 'eval':
            active_top_k = (lengths * self.args.pi).ceil().long()
#            print(z)
#            if self.args.pi_int > 0:
#                active_top_k = torch.ones(lengths.size()).long().to(device) * self.args.pi_int

            # this is essentially sorting from large to small
            _, z_hard_inds = z.topk(z.size(-1), dim=-1)
            z_hard = torch.zeros(z.size()).long().to(device)

            # TODO: must be a better way than using a loop here
            for i in range(z.size(0)):
                subidx = z_hard_inds[i, :active_top_k[i]]
                z_hard[i, subidx] = 1.0
            z = z_hard
#            z = z.scatter(1, gold_sent_positions.unsqueeze(1), 1.0)
#            print(z)
#            input()
        return z

    # tensor of booleans 
    def create_sent_mask(self, sent_lengths, max_sent_length):
        max_sent_length = max_sent_length if self.args.dataparallel else sent_lengths.max().item()
        return torch.arange(max_sent_length).expand(sent_lengths.size(0), max_sent_length).to(sent_lengths.device) < sent_lengths.unsqueeze(1)
    
    def convert_z_from_sentence_to_token_level(self, z, sent_starts, sent_ends, fix_positions, attention_mask):
        device = attention_mask.device
        mask = []
        sent_token_lengths = sent_ends - sent_starts
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
        kl_loss = kl_loss.sum(dim=1) / mask.sum(dim=1)
        return kl_loss.mean()
    
    def calc_sent_rationale_loss(self, sent_logits, gold_sent_positions, sent_mask):
        device = sent_logits.device
        rationale = torch.zeros(sent_mask.size()).to(device).scatter(1, gold_sent_positions.unsqueeze(1), 1.0)
        active_loss_pos = (sent_mask.view(-1) == 1)
        active_logits = sent_logits.view(-1)[active_loss_pos]
        active_labels = rationale.float().view(-1)[active_loss_pos]
        sent_rationale_loss = F.binary_cross_entropy_with_logits(active_logits, active_labels)
        return sent_rationale_loss

    def forward_eval(self, batch):
        with torch.no_grad():
            return self.forward(batch, mode='eval')


class SentenceLevelEncoder(nn.Module):
    def __init__(self, args):
        super(SentenceLevelEncoder, self).__init__()
        self.args = args

        encoder_model_class = get_encoder_model_class(args)
        self.model = encoder_model_class.from_pretrained(args.encoder_type, cache_dir=self.args.cache_dir)

    def forward(self, input_ids, attention_mask, sent_starts, sent_ends):
        token_reps = self.model(input_ids, attention_mask)[0]
        sent_reps = self.extract_sentence_reps(token_reps, sent_starts, sent_ends)
        return sent_reps, token_reps
    
    def extract_sentence_reps(self, token_reps, sent_starts, sent_ends):
        batch_size, _, hidden_size = token_reps.size()
        num_sents = sent_starts.size(1)
        sent_rep_shape = (batch_size, num_sents, hidden_size)
        
        sent_start_reps = token_reps.gather(dim=1, index=sent_starts.unsqueeze(-1).expand(sent_rep_shape))
        sent_end_reps = token_reps.gather(dim=1, index=sent_ends.unsqueeze(-1).expand(sent_rep_shape))
        sent_reps = torch.cat((sent_start_reps, sent_end_reps), dim=-1)
        # print(f'shape token reps: {token_reps.shape}') # 1, 128, 768
        # print(f'shape sentence reps: {sent_reps.shape}')  # 1, 2, 1536
        return sent_reps


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args

        self.model = BertForSequenceClassification.from_pretrained(
            args.decoder_type,
            num_labels=2, #3
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_dir=self.args.cache_dir,
        )
    
    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output


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
        gold_sent_positions = batch['gold_sent_mask']

        sent_reps, token_reps = self.encoder(input_ids, attention_mask, sent_starts, sent_ends)
        # print(f'sent_reps: {sent_reps}')
        # print(f'token_reps: {token_reps}')

        batch_size, num_tokens, _ = token_reps.size() # 1, 128
        sent_logits = self.rep_to_logit_layer(sent_reps)
        sent_logits = self.drop(sent_logits)
        sent_logits = sent_logits.squeeze(2)
        sent_mask = self.create_sent_mask(sent_lengths, batch['max_sent_length']).float()
        # print(f'sent_logits: {sent_logits}')
        # print(f'sent_mask: {sent_mask}')

        sent_z = self.sample_z(sent_logits, sent_mask, mode)   ## sampling 
        # print(f'sent_z: {sent_z}') ## intermediate representation

        ### new attention mask
        ### fix_positions b/c we want the query to always be masked to 1
        token_z = self.convert_z_from_sentence_to_token_level(
            sent_z,
            sent_starts,
            sent_ends,
            fix_positions,
            attention_mask
        )
        
        # print(f'input_ids: {input_ids}')
        # print(f'token_z: {token_z}')
        # print(f'labels: {labels}')

        output = self.decoder(
            input_ids,
            token_z,
            labels
        )
        # print(f'output: {output}')

        kl_loss = self.calc_kl_loss(sent_logits, sent_mask)
        loss = output.loss + self.args.beta * kl_loss

        if self.args.use_gold_rationale:
            gold_sent_rationale_loss = self.calc_sent_rationale_loss(sent_logits, gold_sent_positions, sent_mask)
            loss += self.args.gamma * gold_sent_rationale_loss
        else:
            gold_sent_rationale_loss = 0

        if self.args.dataparallel:
            loss = loss.item()
            kl_loss = kl_loss.item()
            # gold_sent_rationale_loss = torch.tensor([gold_sent_rationale_loss])

        return {
            'loss': loss,
            'pred_loss': output.loss,
            'logits': output.logits,
            'kl_loss': kl_loss,
            'token_z': token_z,
            'sent_z': sent_z,
            # 'gold_sent_rationale_loss': gold_sent_rationale_loss,
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
        # rationale = torch.zeros(sent_mask.size()).to(device).scatter(1, gold_sent_positions.unsqueeze(1), 1.0)
        active_loss_pos = (sent_mask.view(-1) == 1)
        active_logits = sent_logits.view(-1)[active_loss_pos]
        active_labels = rationale.float().view(-1)[active_loss_pos]
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
