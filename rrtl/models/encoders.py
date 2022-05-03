import torch
import torch.nn as nn
import torch.nn.functional as F

from rrtl.transformer_models.modeling_bert import (
    BertModel,
    BertForSequenceClassification,
    BertForQuestionAnswering,
)

from rrtl.transformer_models.modeling_distilbert import (
    DistilBertModel,
    DistilBertForSequenceClassification,
    DistilBertForQuestionAnswering,
)

from rrtl.transformer_models.modeling_roberta import (
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaForQuestionAnswering,
)


def get_encoder_model_class(args):
    if args.encoder_type == 'bert-base-uncased':
        return BertModel
    elif args.encoder_type == 'distilbert-base-uncased':
        return DistilBertModel
    elif args.encoder_type == 'roberta-large':
        return RobertaModel


class SentenceLevelEncoder(nn.Module):
    def __init__(self, args):
        super(SentenceLevelEncoder, self).__init__()
        self.args = args

        encoder_model_class = get_encoder_model_class(args)
        self.model = encoder_model_class.from_pretrained(args.encoder_type, cache_dir=args.cache_dir)

    def forward(self, input_ids, attention_mask, sent_starts, sent_ends):
        token_reps = self.model(input_ids, attention_mask)[0]
        sent_reps = self.extract_sentence_reps(token_reps, sent_starts, sent_ends)
        return sent_reps, token_reps
    
    def extract_sentence_reps(self, token_reps, sent_starts, sent_ends):
        batch_size, _, hidden_size = token_reps.size()
        num_sents = sent_starts.size(1)
        sent_rep_shape = (batch_size, num_sents, hidden_size)

        nonzeros = (sent_ends > 0)
        sent_ends_ = sent_ends.clone().detach()
        sent_ends_[nonzeros] -= 1

        sent_start_reps = token_reps.gather(dim=1, index=sent_starts.unsqueeze(-1).expand(sent_rep_shape))
        sent_end_reps = token_reps.gather(dim=1, index=sent_ends_.unsqueeze(-1).expand(sent_rep_shape))
        sent_reps = torch.cat((sent_start_reps, sent_end_reps), dim=-1)
        return sent_reps


class TokenLevelEncoder(nn.Module):
    def __init__(self, args):
        super(TokenLevelEncoder, self).__init__()
        self.args = args
        encoder_model_class = get_encoder_model_class(args)
        self.model = encoder_model_class.from_pretrained(args.encoder_type, cache_dir=args.cache_dir)

    def forward(self, input_ids, attention_mask):
        token_reps = self.model(input_ids, attention_mask)[0]
        return token_reps
