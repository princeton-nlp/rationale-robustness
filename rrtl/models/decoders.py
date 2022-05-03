import torch
import torch.nn as nn
import torch.nn.functional as F

from rrtl.transformer_models.modeling_bert import (
    BertModel,
    BertForSequenceClassification,
    BertForQuestionAnswering,
)
#from transformers import BertForSequenceClassification

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

from rrtl.models.encoders import get_encoder_model_class


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
    elif args.decoder_type == 'roberta-large':
        return {
            'classification': RobertaForSequenceClassification, 
            'qa': RobertaForQuestionAnswering,
        }


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args

        self.model = BertForSequenceClassification.from_pretrained(
            args.decoder_type,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_dir=self.args.cache_dir,
        )
    
    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output


class ClassificationDecoder(nn.Module):
    def __init__(self, args):
        super(ClassificationDecoder, self).__init__()
        self.args = args

        encoder_model_class = get_encoder_model_class(args)
        self.model = encoder_model_class.from_pretrained(
            args.decoder_type,
            num_labels=3,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            cache_dir=self.args.cache_dir,
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
            output_hidden_states=True,
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