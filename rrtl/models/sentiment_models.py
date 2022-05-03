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


class FullContextSentimentModel(nn.Module):
    def __init__(self, args):
        super(FullContextSentimentModel, self).__init__()
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
        rationales = batch['rationales']

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

#        fix_mask = (torch.arange(logits.size(1)).expand(batch_size, logits.size(1)).to(device) < fix_positions.unsqueeze(1)).long()

        if mode == 'train':
            relaxed_bernoulli_dist = RelaxedBernoulli(self.args.tau, logits=logits)
            z = relaxed_bernoulli_dist.rsample()
#            z = z.masked_fill(fix_mask.bool(), 1.0)
        elif mode == 'eval':
            z = torch.sigmoid(logits)
        # mask out paddings
        z = z.masked_fill(~mask.bool(), 0.0)

        if mode == 'eval':
            #z = z.masked_fill(fix_mask.bool(), 1.0)
            #active_top_k = ((lengths - fix_positions.sum(-1)) * self.args.pi).ceil().long()
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
