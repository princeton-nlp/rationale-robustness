import os
import csv
import re
import string
import random
import json
from pprint import pprint

from tqdm import tqdm
import pandas as pd
from nltk.corpus import words
from nltk.tokenize import sent_tokenize

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, RobertaTokenizerFast

from rrtl.config import Config

config = Config()
random.seed(0)

PRINTABLE = set(string.printable)


def get_special_token_map(encoder_type):
    if encoder_type.startswith('roberta'):
        special_token_map = {
            'bos_token': '<s>',
            'eos_token': '</s>',
            'sep_token': '</s>',
            'cls_token': '<s>',
            'unk_token': '<unk>',
            'pad_token': '<pad>',
            'mask_token': '<mask>',
        }
    elif encoder_type.startswith('bert') or encoder_type.startswith('distilbert'):
        special_token_map = {
            'sep_token': '[SEP]',
            'cls_token': '[CLS]',
            'unk_token': '[UNK]',
            'pad_token': '[PAD]',
            'mask_token': '[MASK]',
        }
    return special_token_map


def extract_token_rationales(text, highlight_text, highlight_mode='pos'):
    """
    Take text with gold rationale highlighted as *word*.
    The function also extract neg rationale annotated as [word] when `highlight_mode` = 'neg'.
    """
    # get char-level spans for rationales and words
    if highlight_mode == 'pos':
        left_sym = '*'
        right_sym = '*'
    elif highlight_mode == 'neg':
        left_sym = '['
        right_sym = ']'
    pattern_string = f'[\{left_sym}]+.*?[\{right_sym}]+'

    highlight_matches = [m for m in re.finditer(pattern_string, highlight_text)]
    highlight_spans = [m.span() for m in highlight_matches]
    if not highlight_spans:
        return []

    highlight_words = [highlight_text[s:e].strip(string.punctuation) for s, e in highlight_spans]
    pattern = '.*'.join([f'({word})' for word in highlight_words])
    matches = [m for m in re.finditer(pattern, text)]
    if len(matches) != 1:
        # same highlight word appears many times
        # match the highlight words again in the highlight text
        # check if there are surrounding regex symbols to decide which appearance to choose
        selector = [
            True if highlight_text[s - 1:s] == left_sym or highlight_text[e:e + 1] == right_sym else False 
            for s, e in [m.span() for m in re.finditer(pattern, highlight_text)]
        ]
        matches = [m for m, is_select in zip(matches, selector) if is_select == True]

    if not matches:
        return []
    # most common case should be only one match
    match = matches[0]

    spans = [list(match.span(i)) for i in range(1, len(match.groups()) + 1)]
    assert ' '.join(highlight_words) == ' '.join([text[s:e] for s, e in spans])
    return spans


def get_rationale_vector_from_spans(offsets, span_set):
    spans = list(span_set)
    span_starts = set([s for s, e in spans])
    span_ends = set([e for s, e in spans])
    start_to_end_map = {s: e for s, e in spans}
    rationale = [0] * len(offsets)
    current_start = float('inf')
    current_end = float('-inf')

    for i, offset in enumerate(offsets):
        offset_start, offset_end = offset
        if offset_start in span_starts:
            rationale[i] = 1
            current_start = offset_start
            current_end = start_to_end_map[offset_start]
        elif offset_end in span_ends:
            rationale[i] = 1
            current_start = float('inf')
            current_end = float('-inf')
        elif offset_start > current_start and offset_end < current_end:
            rationale[i] = 1
    return rationale


def get_fixed_masks(tokenized, tokenizer):
    s_pos = []
    tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0].tolist())

    for pos, token in enumerate(tokens):
        if token in ('[CLS]', '[SEP]'):
            s_pos.append(pos)
        if token == '[PAD]':
            continue
    if len(s_pos) != 3:
        return None, None, None
    cls_pos, sep1_pos, sep2_pos = s_pos
    p_start_end = [cls_pos + 1, sep1_pos - 1]
    h_start_end = [sep1_pos + 1, sep2_pos - 1]
    return s_pos, p_start_end, h_start_end


class BaseDataLoader:
    def __init__(self, args):
        self.args = args
        self.tok_kwargs = config.TOK_KWARGS
        self.tok_kwargs['max_length'] = self.args.max_length
        if self.args.encoder_type.startswith('bert') or self.args.encoder_type.startswith('distilbert'):
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=self.args.cache_dir)
        elif self.args.encoder_type.startswith('roberta'):
            self.tokenizer = RobertaTokenizerFast.from_pretrained(self.args.encoder_type, cache_dir=self.args.cache_dir)

        self.dataset_name_to_dataset_class = {
            'squad': SQUADDataset,
            'squad-nr': SQUADNegRationaleDataset,
            'squad-addonesent': SQUADDataset,
            'squad-addonesent-pos0': SQUADDataset,
            'squad-addrand-pos0': SQUADDataset,
            'squad-addwiki-pos0': SQUADDataset,
            'squad-addrand': SQUADDataset,
            'squad-addwiki': SQUADDataset,
            'squad-addsent': SQUADDataset,
            'fever': FeverDataset,
            'multirc': MultiRCDataset,
            'beer': SentimentDataset,
            'hotel': SentimentDataset,
        }
        self._dataloaders = {}
        self.special_token_map = get_special_token_map(self.args.encoder_type)

    def _load_processed_data(self, mode):
        raise NotImplementedError

    def _build_dataloader(self, data, mode):
        dataset = self.dataset_name_to_dataset_class[self.args.dataset_name](
            self.args,
            data,
            self.tokenizer,
            self.tok_kwargs
        )
        collate_fn = dataset.collater
        batch_size = self.args.batch_size
        shuffle = (not self.args.no_shuffle) if mode == 'train' else False

        self._dataloaders[mode] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
        print(f'[{mode}] dataloader built => {len(dataset)} examples')
    
    def build(self, mode):
        data = self._load_processed_data(mode)
        self._build_dataloader(data, mode)

    def build_all(self):
        for mode in ['train', 'dev', 'test']:
            self.build(mode)
    
    def __getitem__(self, mode):
        return self._dataloaders[mode]

    @property
    def train(self):
        return self._dataloaders['train']

    @property
    def dev(self):
        return self._dataloaders['dev']
    
    @property
    def test(self):
        return self._dataloaders['test']


class BaseDataset(Dataset):
    def __init__(self, args, data, tokenizer, tok_kwargs):
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.tok_kwargs = tok_kwargs

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    @property
    def num_batches(self):
        return len(self.data) // self.args.batch_size


class SQUADDataLoader(BaseDataLoader):
    def __init__(self, args):
        super(SQUADDataLoader, self).__init__(args)
        if self.args.dataset_name == 'squad' and self.args.dataset_split == 'all':
            self.build('train')
            self.build('dev')
        elif self.args.dataset_name == 'squad':
            self.build('dev')
        else:
            self.build_adv_squad('dev')
        
    def build_adv_squad(self, mode):
        example_ids = self.get_squad_dev_ids()
        data = self._load_processed_data(mode, example_ids)
        self._build_dataloader(data, mode)
    
    def get_squad_dev_ids(self):
        example_ids = set()
        path = config.DATA_DIR / 'squad/dev-v1.1.json'
        with open(path) as f:
            obj = json.load(f)
            for data in obj['data']:
                for paragraph in data['paragraphs']:
                    context = paragraph['context']
                    qas = paragraph['qas']
                    for qa in qas:
                        example_id = qa['id']
                        example_ids.add(example_id)
        return example_ids

    def load_raw_data(self, mode, example_ids_to_skip=None):
        datapoints = []
        if self.args.dataset_name == 'squad':
            path = config.DATA_DIR / f'squad/{mode}-v1.1.json'
        elif self.args.dataset_name == 'squad-addsent':
            path = config.DATA_DIR / f'squad/adv/sample1k-HCVerifyAll.json'
        elif self.args.dataset_name == 'squad-addonesent':
            path = config.DATA_DIR / f'squad/adv/sample1k-HCVerifySample.json'
        elif self.args.dataset_name == 'squad-addonesent-pos0':
            path = config.DATA_DIR / f'squad/adv/sample1k-HCVerifySample-pos0.jsonl'
        elif self.args.dataset_name == 'squad-addrand-pos0':
            path = config.DATA_DIR / f'squad/adv/addrand_pos0.jsonl'
        elif self.args.dataset_name == 'squad-addrand':
            path = config.DATA_DIR / f'squad/adv/addrand.jsonl'
        elif self.args.dataset_name == 'squad-addwiki-pos0':
            path = config.DATA_DIR / f'squad/adv/addwiki_pos0.jsonl'
        elif self.args.dataset_name == 'squad-addwiki':
            path = config.DATA_DIR / f'squad/adv/addwiki.jsonl'
        else:
            raise ValueError(f'Dataset {self.args.dataset_name} not supported.')

        if self.args.dataset_name in ('squad', 'squad-addsent', 'squad-addonesent'):
            with open(path) as f:
                obj = json.load(f)
                for data in obj['data']:
                    for paragraph in data['paragraphs']:
                        context = paragraph['context'].strip()
                        qas = paragraph['qas']
                        for qa in qas:
                            example_id = qa['id']
                            question = qa['question'].strip()
                            if example_ids_to_skip is not None and example_id in example_ids_to_skip:
                                continue
                            answer = qa['answers'][0]
                            answer_texts = [answer['text'] for answer in qa['answers']]
                            start = answer['answer_start']
                            end = answer['answer_start'] + len(answer['text'])

                            datapoints.append({
                                'example_id': example_id,
                                'context': context,
                                'question': question,
                                'start': start,
                                'end': end,
                                'answer_texts': answer_texts,
                            })
        elif self.args.dataset_name in ('squad-addonesent-pos0', 'squad-addrand-pos0', 'squad-addrand', 'squad-addwiki-pos0', 'squad-addwiki'):
            with open(path) as f:
                for line in f:
                    datapoint = json.loads(line)
                    datapoints.append(datapoint)

        if self.args.debug:
            datapoints = datapoints[:200]
        return datapoints
    
    def _load_processed_data(self, mode, example_ids=None):
        datapoints = self.load_raw_data(mode, example_ids)
        processed_datapoints = []

        print('Data preprocessing...')
        for datapoint in tqdm(datapoints, total=len(datapoints)):
            example_id = datapoint['example_id']
            context = datapoint['context'].strip()
            question = datapoint['question'].strip()
            answer_texts = datapoint['answer_texts']
            sep = ' ' + self.special_token_map['sep_token'] + ' '
            input_text = question + sep + context
            tokenized = self.tokenizer.encode_plus(input_text, **self.tok_kwargs)
            input_ids = tokenized['input_ids'][0]
            attention_mask = tokenized['attention_mask'][0]
            offsets = tokenized['offset_mapping'][0].tolist()
            start = datapoint['start'] + len(question + sep)
            end = datapoint['end'] + len(question + sep)

            orig_tokens_with_space = self.get_orig_tokens_with_space(input_text, offsets)
            span_start, span_end = self.get_answer_span(offsets, start, end, orig_tokens_with_space, answer_texts)

            tokens = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0].tolist())
            question_end_position = tokens.index(self.special_token_map['sep_token']) + 1

            context_sents = sent_tokenize(context)
            sent_starts, sent_ends = self.get_sent_start_end(orig_tokens_with_space, context_sents)

            if ''.join(orig_tokens_with_space[span_start:span_end]).strip() not in answer_texts:
                continue

            if len(sent_starts) != len(context_sents):
                continue
            
            gold_sent_pos = self.get_gold_sent_pos(sent_starts, sent_ends, span_start, span_end)
            if gold_sent_pos < 0:
                continue

            sent_length = len(sent_starts)
            
            label = [span_start, span_end]
            gold_answers = answer_texts
            token_offsets = offsets
            fix_position = question_end_position

            question_input_ids = input_ids[:question_end_position]
            context_input_ids = input_ids.detach().clone()
            context_input_ids[:question_end_position] = 0  # [PAD]
            context_input_ids[question_end_position - 1] = 101  # [CLS]
            context_input_ids = torch.roll(context_input_ids, -question_end_position + 1)
            assert context_input_ids[0].item() == 101

            processed_datapoints.append({
                'example_id': example_id,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label,
                'input_text': input_text,
                'gold_answers': answer_texts,
                'token_offsets': offsets,
                'sent_starts': [0] + sent_starts,
                'sent_ends': [sent_starts[0]] + sent_ends,
                'sent_length': sent_length,  # this is noly the context sentences
                'question_end_position': question_end_position,
                'gold_sent_pos': gold_sent_pos,
                'context_sents': context_sents,
                'question_text': question,
                'context_text': context,
                'question_input_ids': question_input_ids,
                'context_input_ids': context_input_ids,
            })
        return processed_datapoints

    def get_orig_tokens_with_space(self, input_text, offsets):
        orig_tokens = []
        for i in range(len(offsets) - 1):
            s, e = offsets[i]
            n_s, n_e = offsets[i + 1]
            if n_s == e + 1:
                orig_tokens.append(input_text[s:e] + ' ')
            elif n_s == e:
                orig_tokens.append(input_text[s:e])
            elif e != 0 and n_s == 0 and n_e == 0:
                orig_tokens.append(input_text[s:e])
                break
            else:
                orig_tokens.append(input_text[s:n_s])
        return orig_tokens
    
    def get_answer_span(self, offsets, start, end, orig_tokens_with_space, answer_texts):
        span_start = -1
        span_end = -1
        for i, (s, e) in enumerate(offsets):
            if start >= s:
                span_start = i
            if end <= e:
                span_end = i
            if span_start != -1 and span_end != -1:
                break
        return span_start, span_end + 1
    
    def get_sent_start_end(self, orig_tokens_with_space, context_sents):
        sep_pos = [i for i, orig_token in enumerate(orig_tokens_with_space) if orig_token.strip() == self.special_token_map['sep_token']]
        assert len(sep_pos) == 1
        sep_pos = sep_pos[0]

        sent_starts = [sep_pos + 1]
        sent_ends = []
        curr_context_sent_index = 0
        curr_accu_sent = ''

        for i, orig_token in enumerate(orig_tokens_with_space):
            if i <= sep_pos:
                continue
            curr_context_sent = context_sents[curr_context_sent_index]
            curr_accu_sent += orig_token
            if curr_accu_sent.strip() == curr_context_sent:
                sent_starts.append(i + 1)
                sent_ends.append(i + 1)
                curr_context_sent_index += 1
                curr_accu_sent = ''
        sent_starts = sent_starts[:-1]
        return sent_starts, sent_ends

    def normalize(self, text):
        text = re.sub(' +', ' ', text)
        text = re.sub('\n', ' ', text)
        return text
    
    def get_gold_sent_pos(self, sent_starts, sent_ends, span_start, span_end):
        gold_sent_pos = -1
        for i, (sent_start, sent_end) in enumerate(zip(sent_starts, sent_ends)):
            if span_start >= sent_start and span_end <= sent_end:
                gold_sent_pos = i
                break
        return gold_sent_pos


class SQUADDataset(BaseDataset):
    def __init__(self, args, data, tokenizer, tok_kwargs):
        super(SQUADDataset, self).__init__(args, data, tokenizer, tok_kwargs)

    def collater(self, batch):
        device = 'cuda' if self.args.use_cuda else 'cpu'
        question_input_ids = torch.tensor(self.pad([datapoint['question_input_ids'].tolist() for datapoint in batch])).long().to(device)
        question_attn_mask = (question_input_ids > 0).long()
        context_input_ids = torch.tensor(self.pad([datapoint['context_input_ids'].tolist() for datapoint in batch])).long().to(device)
        context_attn_mask = (context_input_ids > 0).long()
        return {
            'example_id': [datapoint['example_id'] for datapoint in batch],
            'input_ids': torch.stack([datapoint['input_ids'] for datapoint in batch]).to(device),
            'attention_mask': torch.stack([datapoint['attention_mask'] for datapoint in batch]).to(device),
            'labels': torch.tensor([datapoint['label'] for datapoint in batch]).long().to(device),
            'input_texts': [datapoint['input_text'] for datapoint in batch],
            'gold_answers': [datapoint['gold_answers'] for datapoint in batch],
            'token_offsets': [datapoint['token_offsets'] for datapoint in batch],
            'sent_starts': torch.tensor(self.pad([datapoint['sent_starts'] for datapoint in batch])).long().to(device),
            'sent_ends': torch.tensor(self.pad([datapoint['sent_ends'] for datapoint in batch])).to(device),
            'sent_lengths': torch.tensor([datapoint['sent_length'] for datapoint in batch]).to(device),
            'question_end_positions': torch.tensor([datapoint['question_end_position'] for datapoint in batch]).to(device),
            'gold_sent_positions': torch.tensor([datapoint['gold_sent_pos'] for datapoint in batch]).to(device),
            'max_sent_length': max([datapoint['sent_length'] for datapoint in batch]),
            'context_sents': [datapoint['context_sents'] for datapoint in batch],
            'question_text': [datapoint['question_text'] for datapoint in batch],
            'context_text': [datapoint['context_text'] for datapoint in batch],
            'question_input_ids': question_input_ids,
            'question_attn_mask': question_attn_mask,
            'context_input_ids': context_input_ids,
            'context_attn_mask': context_attn_mask,
        }

    def pad(self, seqs):
        padded_seqs = []
        max_length = max(len(seq) for seq in seqs)
        return [seq + (max_length - len(seq)) * [0] for seq in seqs]


class FeverDataLoader(BaseDataLoader):
    def __init__(self, args):
        super(FeverDataLoader, self).__init__(args)
        if args.dataset_split == 'all':
            self.build_all()
        else:
            self.build(args.dataset_split)

    def _load_raw_data(self, mode):
        datapoints = []
        if self.args.attack_path is not None:
            path = os.path.join(self.args.attack_path, 'val.jsonl')
        else:
            path = f'{config.PROJECT_DIR}/data/fever/{mode}.jsonl'
        with open(path, 'r') as f:
            for line in f:
                content = json.loads(line)
                label = config.FEVER_LABEL[content['classification']]
                question = content['query']
                evidences = content['evidences']
                docids = []
                for evidence in evidences:
                    evidence = evidence[0]

                    docids.append({
                        'docid': evidence['docid'],
                        'start_token': evidence['start_token'],
                        'end_token': evidence['end_token'],
                        'start_sentence': evidence['start_sentence'],
                        'end_sentence': evidence['end_sentence'],
                        'rationale_text': evidence['text']
                    })
                datapoints.append({
                    'label': label,
                    'question': question,
                    'docids': docids,
                })
        if self.args.debug:
            datapoints = datapoints[:300]
        return datapoints

    def _get_context(self, docids):
        if self.args.attack_path is not None:
            docs_dir = os.path.join(self.args.attack_path, 'docs')
        else:
            docs_dir = f'{config.PROJECT_DIR}/data/fever/docs'
        context = ''

        # all evidences should be from the same docid
        docid_dir = docids[0]['docid']

        with open(os.path.join(docs_dir, docid_dir), 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            lines = list(filter(lambda x: bool(len(x)), lines))
            tokenized = [list(filter(lambda x: bool(len(x)), line.strip().split(' '))) for line in lines]
            context = context + ' '.join(lines)

        return context

    def _load_processed_data(self, mode):
        datapoints = self._load_raw_data(mode)
        processed_datapoints = []
        for datapoint in tqdm(datapoints, total=len(datapoints)):
            label = datapoint['label']
            question = datapoint['question']
            context = self._get_context(datapoint['docids'])
            
            sep = ' ' + self.special_token_map['sep_token'] + ' '
            input_text = question + sep + context
            tokenized = self.tokenizer.encode_plus(input_text, **self.tok_kwargs)
            input_ids = tokenized['input_ids'][0]
            attention_mask = tokenized['attention_mask'][0]

            offsets = tokenized['offset_mapping'][0].tolist()
            gold_sent_pos = []
            tokens = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0].tolist())
            fix_position = tokens.index(self.special_token_map['sep_token']) + 1

            orig_tokens_with_space = self.get_orig_tokens_with_space(input_text, offsets)
            context_sents = sent_tokenize(context)
            bad_state, sent_starts, sent_ends = self.get_sent_start_end(orig_tokens_with_space, context_sents)
            assert(len(sent_starts) == len(sent_ends))
            gold_answers = []
            gold_sent_pos_start = []
            gold_sent_pos_end = []
            sent_length = len(sent_starts)
            for docid in datapoint['docids']: # iterate over evidences
                gold_answers.append(docid['rationale_text'])
                gold_sent_pos_start.append(docid['start_sentence'])  
                gold_sent_pos_end.append(docid['end_sentence'])
                if (gold_sent_pos_start[-1] >= sent_length):
                    bad_state = True

            gold_sent_mask = [0] * len(input_ids)
            if (not bad_state):
                gold_sent_mask = self.get_gold_sent_mask(gold_sent_mask, orig_tokens_with_space, gold_answers, gold_sent_pos_start, gold_sent_pos_end, sent_starts, sent_ends)

            if (not bad_state):
                processed_datapoints.append({
                        'input_ids': input_ids, # ok
                        'input_text': input_text, # ok
                        'attention_mask': attention_mask, # ok
                        'label': label, # ok
                        'fix_position': fix_position,
                        # 'gold_sent_mask': gold_sent_mask,
                        'gold_sent_pos': gold_sent_pos_start, 
                        'sent_starts': [0] + sent_starts,
                        'sent_ends': [sent_starts[0]] + sent_ends,
                        'sent_length': sent_length,
                })
        return processed_datapoints

    def get_gold_sent_mask(self, gold_sent_mask, orig_tokens_with_space, gold_answers, gold_sent_pos_start, gold_sent_pos_end, sent_starts, sent_ends):
        for i, start in enumerate(gold_sent_pos_start):
            end = gold_sent_pos_end[i]

            if (start >= len(sent_starts)):
                new_start = sent_starts[-1]
                new_end = sent_ends[-1]
            else:
                new_start = sent_starts[start]
                new_end = sent_ends[end-1]

            gold_sent_mask[new_start:new_end] = [1]*(new_end-new_start)
        return gold_sent_mask

    def get_orig_tokens_with_space(self, input_text, offsets):
        orig_tokens = []
        for i in range(len(offsets) - 1):
            s, e = offsets[i]
            n_s, n_e = offsets[i + 1]
            if n_s == e + 1:
                orig_tokens.append(input_text[s:e] + ' ')
            elif n_s == e:
                orig_tokens.append(input_text[s:e])
            elif e != 0 and n_s == 0 and n_e == 0:
                orig_tokens.append(input_text[s:e])
                break
            else:
                orig_tokens.append(input_text[s:n_s])
        return orig_tokens
    
    def get_sent_start_end(self, orig_tokens_with_space, context_sents):
        bad_example = False
        sep_pos = [i for i, orig_token in enumerate(orig_tokens_with_space) if orig_token.strip() == '[SEP]']
        assert len(sep_pos) == 1
        sep_pos = sep_pos[0]

        sent_starts = [sep_pos + 1]

        sent_ends = []
        curr_context_sent_index = 0
        curr_accu_sent = ''

        for i, orig_token in enumerate(orig_tokens_with_space):
            if i <= sep_pos:
                continue
            curr_context_sent = context_sents[curr_context_sent_index]
            curr_accu_sent += orig_token

            if curr_accu_sent.strip() == curr_context_sent:
                sent_starts.append(i + 1)
                sent_ends.append(i + 1)
                curr_context_sent_index += 1
                curr_accu_sent = ''

        if (len(sent_starts) == 1):
            bad_example = True
        sent_starts = sent_starts[:-1]  

        return bad_example, sent_starts, sent_ends

    def normalize(self, text):
        text = re.sub(' +', ' ', text)
        text = re.sub('\n', ' ', text)
        return text

    def get_gold_sent_pos(self, sent_starts, sent_ends, span_start, span_end):
        gold_sent_pos = -1
        for i, (sent_start, sent_end) in enumerate(zip(sent_starts, sent_ends)):
            if span_start >= sent_start and span_end <= sent_end:
                gold_sent_pos = i
                break
        return gold_sent_pos


class FeverDataset(BaseDataset):
    def __init__(self, args, data, tokenizer, tok_kwargs):
        super(FeverDataset, self).__init__(args, data, tokenizer, tok_kwargs)

    def collater(self, batch):
        device = 'cuda' if self.args.use_cuda else 'cpu'
        return {
            'input_ids': torch.stack([datapoint['input_ids'] for datapoint in batch]).to(device),
            'attention_mask': torch.stack([datapoint['attention_mask'] for datapoint in batch]).to(device),
            'labels': torch.tensor([datapoint['label'] for datapoint in batch]).long().to(device),
            'sent_starts': torch.tensor(self.pad([datapoint['sent_starts'] for datapoint in batch])).long().to(device),
            'sent_ends': torch.tensor(self.pad([datapoint['sent_ends'] for datapoint in batch])).to(device),
            'sent_lengths': torch.tensor([datapoint['sent_length'] for datapoint in batch]).to(device),
            'fix_positions': torch.tensor([datapoint['fix_position'] for datapoint in batch]).to(device),
            # 'gold_sent_mask': torch.tensor(self.pad([datapoint['gold_sent_mask'] for datapoint in batch])).long().to(device),
            'gold_sent_positions': torch.tensor(self.pad_gold([datapoint['gold_sent_pos'] for datapoint in batch])).to(device),
            'max_sent_length': max([datapoint['sent_length'] for datapoint in batch]),
        }

    def pad(self, seqs):
        padded_seqs = []
        max_length = max(len(seq) for seq in seqs)
        return [seq + (max_length - len(seq)) * [0] for seq in seqs]

    def pad_gold(self, seqs):
        padded_seqs = []
        max_length = max(len(seq) for seq in seqs)
        return [seq + (max_length - len(seq)) * [seq[0]] for seq in seqs]


class MultiRCDataLoader(BaseDataLoader):
    def __init__(self, args):
        super(MultiRCDataLoader, self).__init__(args)
        if args.dataset_split == 'all':
            self.build_all()
        else:
            self.build(args.dataset_split)

    def _load_raw_data(self, mode):
        datapoints = []
        if self.args.attack_path is not None:
            path = os.path.join(self.args.attack_path, 'val.jsonl')
        else:
            path = f'{config.PROJECT_DIR}/data/multirc/{mode}.jsonl'
        with open(path, 'r') as f:
            for line in f:
                content = json.loads(line)
                label = config.MULTIRC_LABEL[content['classification']]
                question = content['query']
                evidences = content['evidences']
                docids = []
                for evidence in evidences:
                    evidence = evidence[0]

                    docids.append({
                        'docid': evidence['docid'],
                        'start_token': evidence['start_token'],
                        'end_token': evidence['end_token'],
                        'start_sentence': evidence['start_sentence'],
                        'end_sentence': evidence['end_sentence'],
                        'rationale_text': evidence['text']
                    })
                datapoints.append({
                    'label': label,
                    'question': question,
                    'docids': docids,
                })
        if self.args.debug:
            datapoints = datapoints[:200]
        return datapoints

    def _get_context(self, docids):
        if self.args.attack_path is not None:
            docs_dir = os.path.join(self.args.attack_path, 'docs')
        else:
            docs_dir = f'{config.PROJECT_DIR}/data/multirc/docs'
        context = ''

        # all evidences should be from the same docid
        docid_dir = docids[0]['docid']

        with open(os.path.join(docs_dir, docid_dir), 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            lines = list(filter(lambda x: bool(len(x)), lines))
            tokenized = [list(filter(lambda x: bool(len(x)), line.strip().split(' '))) for line in lines]
            context = context + ' '.join(lines)

        return context

    def _load_processed_data(self, mode):
        datapoints = self._load_raw_data(mode)
        processed_datapoints = []
        for datapoint in tqdm(datapoints, total=len(datapoints)):
            label = datapoint['label']
            question = datapoint['question']
            context = self._get_context(datapoint['docids'])
            
            sep = ' ' + self.special_token_map['sep_token'] + ' '
            input_text = question + sep + context
            tokenized = self.tokenizer.encode_plus(input_text, **self.tok_kwargs)
            input_ids = tokenized['input_ids'][0]
            attention_mask = tokenized['attention_mask'][0]

            offsets = tokenized['offset_mapping'][0].tolist()
            gold_sent_pos = []
            tokens = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0].tolist())
            
            fix_position = tokens.index(self.special_token_map['sep_token']) + 1

            orig_tokens_with_space = self.get_orig_tokens_with_space(input_text, offsets)
            context_sents = sent_tokenize(context)
            bad_state, sent_starts, sent_ends = self.get_sent_start_end(orig_tokens_with_space, context_sents)
            assert(len(sent_starts) == len(sent_ends))
            gold_answers = []
            gold_sent_pos_start = []
            gold_sent_pos_end = []
            sent_length = len(sent_starts)
            for docid in datapoint['docids']: # iterate over evidences
                gold_answers.append(docid['rationale_text'])
                gold_sent_pos_start.append(docid['start_sentence'])  
                gold_sent_pos_end.append(docid['end_sentence'])
                if (gold_sent_pos_start[-1] >= sent_length):
                    bad_state = True

            gold_sent_mask = [0] * len(input_ids)
            # print(gold_sent_pos_start)
            # print(gold_sent_pos_end)
            if (not bad_state):
                gold_sent_mask = self.get_gold_sent_mask(gold_sent_mask, orig_tokens_with_space, gold_answers, gold_sent_pos_start, gold_sent_pos_end, sent_starts, sent_ends)
            
            # print(gold_sent_mask)
            if (not bad_state):
                processed_datapoints.append({
                        'input_ids': input_ids, # ok
                        'input_text': input_text, # ok
                        'attention_mask': attention_mask, # ok
                        'label': label, # ok
                        'fix_position': fix_position,
                        # 'gold_sent_mask': (gold_sent_mask if self.args.use_gold_rationale else None),
                        'gold_sent_pos': gold_sent_pos_start, 
                        'sent_starts': [0] + sent_starts,
                        'sent_ends': [sent_starts[0]] + sent_ends,
                        'sent_length': sent_length,
                    }) 
        return processed_datapoints

    def get_gold_sent_mask(self, gold_sent_mask, orig_tokens_with_space, gold_answers, gold_sent_pos_start, gold_sent_pos_end, sent_starts, sent_ends):
        for i, start in enumerate(gold_sent_pos_start):
            end = gold_sent_pos_end[i]

            if (start >= len(sent_starts)):
                new_start = sent_starts[-1]
                new_end = sent_ends[-1]
            else:
                new_start = sent_starts[start]
                new_end = sent_ends[end-1]

            gold_sent_mask[new_start:new_end] = [1]*(new_end-new_start)
        return gold_sent_mask

    def get_orig_tokens_with_space(self, input_text, offsets):
        orig_tokens = []
        for i in range(len(offsets) - 1):
            s, e = offsets[i]
            n_s, n_e = offsets[i + 1]
            if n_s == e + 1:
                orig_tokens.append(input_text[s:e] + ' ')
            elif n_s == e:
                orig_tokens.append(input_text[s:e])
            elif e != 0 and n_s == 0 and n_e == 0:
                orig_tokens.append(input_text[s:e])
                break
            else:
                orig_tokens.append(input_text[s:n_s])
        return orig_tokens
    
    def get_sent_start_end(self, orig_tokens_with_space, context_sents):
        # print(f'orig_tokens_with_space: {orig_tokens_with_space}')
        # print(f'context_sents: {context_sents}')
        bad_example = False
        sep_pos = [i for i, orig_token in enumerate(orig_tokens_with_space) if orig_token.strip() == '[SEP]']
        assert len(sep_pos) == 1
        sep_pos = sep_pos[0]

        sent_starts = [sep_pos + 1]

        sent_ends = []
        curr_context_sent_index = 0
        curr_accu_sent = ''

        for i, orig_token in enumerate(orig_tokens_with_space):
            if i <= sep_pos:
                continue
            curr_context_sent = context_sents[curr_context_sent_index]
            curr_accu_sent += orig_token

            if curr_accu_sent.strip() == curr_context_sent:
                sent_starts.append(i + 1)
                sent_ends.append(i + 1)
                curr_context_sent_index += 1
                curr_accu_sent = ''
        
        # s = sent_starts[0]
        # e = sent_ends[0]
        # print(orig_tokens_with_space[s:e])

        # hacky fix
        # assert(len(sent_starts) > 1)
        if (len(sent_starts) == 1):
            bad_example = True
        sent_starts = sent_starts[:-1]  

        return bad_example, sent_starts, sent_ends

    def normalize(self, text):
        text = re.sub(' +', ' ', text)
        text = re.sub('\n', ' ', text)
        return text
    
    def get_gold_sent_pos(self, sent_starts, sent_ends, span_start, span_end):
        gold_sent_pos = -1
        for i, (sent_start, sent_end) in enumerate(zip(sent_starts, sent_ends)):
            if span_start >= sent_start and span_end <= sent_end:
                gold_sent_pos = i
                break
        return gold_sent_pos


class MultiRCDataset(BaseDataset):
    def __init__(self, args, data, tokenizer, tok_kwargs):
        super(MultiRCDataset, self).__init__(args, data, tokenizer, tok_kwargs)

    def collater(self, batch):
        device = 'cuda' if self.args.use_cuda else 'cpu'
        return {
            'input_ids': torch.stack([datapoint['input_ids'] for datapoint in batch]).to(device),
            'attention_mask': torch.stack([datapoint['attention_mask'] for datapoint in batch]).to(device),
            'labels': torch.tensor([datapoint['label'] for datapoint in batch]).long().to(device),
            'sent_starts': torch.tensor(self.pad([datapoint['sent_starts'] for datapoint in batch])).long().to(device),
            'sent_ends': torch.tensor(self.pad([datapoint['sent_ends'] for datapoint in batch])).to(device),
            'sent_lengths': torch.tensor([datapoint['sent_length'] for datapoint in batch]).to(device),
            'fix_positions': torch.tensor([datapoint['fix_position'] for datapoint in batch]).to(device),
            'gold_sent_positions': torch.tensor(self.pad_gold([datapoint['gold_sent_pos'] for datapoint in batch])).to(device),
            'max_sent_length': max([datapoint['sent_length'] for datapoint in batch]),
        }

    def pad(self, seqs):
        padded_seqs = []
        max_length = max(len(seq) for seq in seqs)
        return [seq + (max_length - len(seq)) * [0] for seq in seqs]

    def pad_gold(self, seqs):
        padded_seqs = []
        max_length = max(len(seq) for seq in seqs)
        return [seq + (max_length - len(seq)) * [seq[0]] for seq in seqs]


class SQUADNegRationaleDataLoader(BaseDataLoader):
    def __init__(self, args):
        super(SQUADNegRationaleDataLoader, self).__init__(args)
        if self.args.dataset_name == 'squad-nr' and self.args.dataset_split == 'all':
            self.build('train')
            self.build('dev')
        elif self.args.dataset_name == 'squad-nr':
            self.build('dev')

    def load_raw_data(self, mode):
        datapoints = []
        if self.args.dataset_name == 'squad-nr' and mode == 'train':
            #path = config.DATA_DIR / f'squad/negative_rationale/train-nr-v1.1.jsonl'
            path = config.DATA_DIR / f'squad/negative_rationale/train-nr-addonesent-v1.1.jsonl'
        elif self.args.dataset_name == 'squad-nr' and mode == 'dev':
            #path = config.DATA_DIR / f'squad/negative_rationale/dev-nr-v1.1.jsonl'
            path = config.DATA_DIR / f'squad/negative_rationale/dev-nr-addonesent-v1.1.jsonl'
        else:
            raise ValueError(f'Dataset {self.args.dataset_name} not supported.')

        with open(path) as f:
            for line in f:
                datapoint = json.loads(line)
                datapoints.append(datapoint)
        if self.args.debug:
            datapoints = datapoints[:200]
        return datapoints
    
    def _load_processed_data(self, mode):
        datapoints = self.load_raw_data(mode)
        processed_datapoints = []

        print('Data preprocessing...')
        for datapoint in tqdm(datapoints, total=len(datapoints)):
            example_id = datapoint['example_id']
            context = datapoint['context'].strip()
            question = datapoint['question'].strip()
            answer_texts = datapoint['answer_texts']
            sep = ' ' + self.special_token_map['sep_token'] + ' '
            input_text = question + sep + context
            tokenized = self.tokenizer.encode_plus(input_text, **self.tok_kwargs)
            input_ids = tokenized['input_ids'][0]
            attention_mask = tokenized['attention_mask'][0]
            offsets = tokenized['offset_mapping'][0].tolist()
            start = datapoint['start'] + len(question + sep)
            end = datapoint['end'] + len(question + sep)
            augs = datapoint['augs']
            insert_positions = datapoint['insert_positions']

            orig_tokens_with_space = self.get_orig_tokens_with_space(input_text, offsets)
            span_start, span_end = self.get_answer_span(offsets, start, end, orig_tokens_with_space, answer_texts)

            tokens = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0].tolist())
            question_end_position = tokens.index(self.special_token_map['sep_token']) + 1

            context_sents = datapoint['sents']
            sent_starts, sent_ends = self.get_sent_start_end(orig_tokens_with_space, context_sents)

            if ''.join(orig_tokens_with_space[span_start:span_end]).strip() not in answer_texts:
                continue

            if len(sent_starts) != len(context_sents):
                continue
            
            gold_sent_pos = self.get_gold_sent_pos(sent_starts, sent_ends, span_start, span_end)
            if gold_sent_pos < 0:
                continue

            sent_length = len(sent_starts)
            
            label = [span_start, span_end]
            gold_answers = answer_texts
            token_offsets = offsets
            fix_position = question_end_position

            processed_datapoints.append({
                'example_id': example_id,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label,
                'input_text': input_text,
                'gold_answers': answer_texts,
                'token_offsets': offsets,
                'sent_starts': [0] + sent_starts,
                'sent_ends': [sent_starts[0]] + sent_ends,
                'sent_length': sent_length,
                'question_end_position': question_end_position,
                'gold_sent_pos': gold_sent_pos,
                'context_sents': context_sents,
                'question_text': question,
                'context_text': context,
                'augs': augs,
                'insert_positions': insert_positions,
            })
        return processed_datapoints

    def get_orig_tokens_with_space(self, input_text, offsets):
        orig_tokens = []
        for i in range(len(offsets) - 1):
            s, e = offsets[i]
            n_s, n_e = offsets[i + 1]
            if n_s == e + 1:
                orig_tokens.append(input_text[s:e] + ' ')
            elif n_s == e:
                orig_tokens.append(input_text[s:e])
            elif e != 0 and n_s == 0 and n_e == 0:
                orig_tokens.append(input_text[s:e])
                break
            else:
                orig_tokens.append(input_text[s:n_s])
        return orig_tokens
    
    def get_answer_span(self, offsets, start, end, orig_tokens_with_space, answer_texts):
        span_start = -1
        span_end = -1
        for i, (s, e) in enumerate(offsets):
            if start >= s:
                span_start = i
            if end <= e:
                span_end = i
            if span_start != -1 and span_end != -1:
                break
        return span_start, span_end + 1
    
    def get_sent_start_end(self, orig_tokens_with_space, context_sents):
        sep_pos = [i for i, orig_token in enumerate(orig_tokens_with_space) if orig_token.strip() == self.special_token_map['sep_token']]
        assert len(sep_pos) == 1
        sep_pos = sep_pos[0]

        sent_starts = [sep_pos + 1]
        sent_ends = []
        curr_context_sent_index = 0
        curr_accu_sent = ''

        for i, orig_token in enumerate(orig_tokens_with_space):
            if i <= sep_pos:
                continue
            curr_context_sent = context_sents[curr_context_sent_index]
            curr_accu_sent += orig_token
            if curr_accu_sent.strip() == curr_context_sent:
                sent_starts.append(i + 1)
                sent_ends.append(i + 1)
                curr_context_sent_index += 1
                curr_accu_sent = ''
        sent_starts = sent_starts[:-1]
        return sent_starts, sent_ends

    def normalize(self, text):
        text = re.sub(' +', ' ', text)
        text = re.sub('\n', ' ', text)
        return text
    
    def get_gold_sent_pos(self, sent_starts, sent_ends, span_start, span_end):
        gold_sent_pos = -1
        for i, (sent_start, sent_end) in enumerate(zip(sent_starts, sent_ends)):
            if span_start >= sent_start and span_end <= sent_end:
                gold_sent_pos = i
                break
        return gold_sent_pos


class SQUADNegRationaleDataset(BaseDataset):
    def __init__(self, args, data, tokenizer, tok_kwargs):
        super(SQUADNegRationaleDataset, self).__init__(args, data, tokenizer, tok_kwargs)

    def collater(self, batch):
        device = 'cuda' if self.args.use_cuda else 'cpu'
        return {
            'example_id': [datapoint['example_id'] for datapoint in batch],
            'input_ids': torch.stack([datapoint['input_ids'] for datapoint in batch]).to(device),
            'attention_mask': torch.stack([datapoint['attention_mask'] for datapoint in batch]).to(device),
            'labels': torch.tensor([datapoint['label'] for datapoint in batch]).long().to(device),
            'input_texts': [datapoint['input_text'] for datapoint in batch],
            'gold_answers': [datapoint['gold_answers'] for datapoint in batch],
            'token_offsets': [datapoint['token_offsets'] for datapoint in batch],
            'sent_starts': torch.tensor(self.pad([datapoint['sent_starts'] for datapoint in batch])).long().to(device),
            'sent_ends': torch.tensor(self.pad([datapoint['sent_ends'] for datapoint in batch])).to(device),
            'sent_lengths': torch.tensor([datapoint['sent_length'] for datapoint in batch]).to(device),
            'question_end_positions': torch.tensor([datapoint['question_end_position'] for datapoint in batch]).to(device),
            'gold_sent_positions': torch.tensor([datapoint['gold_sent_pos'] for datapoint in batch]).to(device),
            'max_sent_length': max([datapoint['sent_length'] for datapoint in batch]),
            'context_sents': [datapoint['context_sents'] for datapoint in batch],
            'question_text': [datapoint['question_text'] for datapoint in batch],
            'context_text': [datapoint['context_text'] for datapoint in batch],
            'insert_positions': torch.tensor([datapoint['insert_positions'] for datapoint in batch]).to(device),
        }

    def pad(self, seqs):
        padded_seqs = []
        max_length = max(len(seq) for seq in seqs)
        return [seq + (max_length - len(seq)) * [0] for seq in seqs]


class SentimentDataLoader(BaseDataLoader):
    def __init__(self, args):
        super(SentimentDataLoader, self).__init__(args)
        if args.dataset_split == 'all':
            self.build_all()
        else:
            self.build(args.dataset_split)

    def _load_raw_data(self, mode):
        datapoints = []
        for aspect in [0]:
            if self.args.attack_path is not None:
                path = self.args.attack_path
            elif self.args.dataset_name == 'beer' and mode in ('train', 'dev'):
                path = config.DATA_DIR / f'sentiment/data/source/beer{aspect}.{mode}'
            elif self.args.dataset_name == 'beer' and mode == 'test':
                path = config.DATA_DIR / 'sentiment/beeradvocate/beer0.test'
            elif self.args.dataset_name == 'hotel' and mode in ('train', 'dev'):
                path = config.DATA_DIR / f'sentiment/data/oracle/hotel_Cleanliness.{mode}'
            elif self.args.dataset_name == 'hotel' and mode == 'test':
                path = config.DATA_DIR / f'sentiment/data/target/hotel_Cleanliness.train'
            else:
                raise ValueError('Dataset name not supported.')
            df = pd.read_csv(path, delimiter='\t')
            for index, row in df.iterrows():
                label = row['label']

                # this could be applied to both beer and hotel
                if label >= 0.6:
                    label = 1  # pos
                elif label <= 0.4:
                    label = 0  # neg
                else:
                    continue
                text = row['text']
                if 'rationale' in row:
                    rationale = [int(r) for r in row['rationale'].split()]
                else:
                    rationale = [-1] * len(row['text'].split())
                datapoints.append({
                    'label': label,
                    'text': text,
                    'rationale': rationale,
                })
        if self.args.debug:
            datapoints = datapoints[:200]
        return datapoints

    def _load_processed_data(self, mode):
        processed_datapoints = []
        datapoints = self._load_raw_data(mode)
        for datapoint in tqdm(datapoints, total=len(datapoints)):
            label = datapoint['label']
            input_tokens = ['[CLS]'] + datapoint['text'].split()
            rationale = [0] + datapoint['rationale']
            input_ids = []
            attention_mask = []
            rationale_ = []
            for input_token, r in zip(input_tokens, rationale):
                tokenized = self.tokenizer.encode_plus(input_token, add_special_tokens=False)
                input_ids += tokenized['input_ids']
                attention_mask += tokenized['attention_mask']
                rationale_ += [r] * len(tokenized['input_ids'])

            if len(input_ids) >= self.args.max_length:
                input_ids = input_ids[:self.args.max_length - 1] + [102]
                attention_mask = attention_mask[:self.args.max_length - 1] + [1]
                rationale = rationale_[:self.args.max_length - 1] + [0]
            else:
                input_ids = input_ids + [102]
                attention_mask = attention_mask + [1]
                rationale = rationale_ + [0]
                
            input_ids = self.pad(input_ids)
            attention_mask = self.pad(attention_mask)
            rationale = self.pad(rationale)

            assert len(input_ids) == self.args.max_length

            processed_datapoints.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label,
                'rationale': rationale,
            })
        return processed_datapoints

    def pad(self, seq):
        return seq + (self.args.max_length - len(seq)) * [0]


class SentimentDataset(BaseDataset):
    def __init__(self, args, data, tokenizer, tok_kwargs):
        super(SentimentDataset, self).__init__(args, data, tokenizer, tok_kwargs)

    def collater(self, batch):
        device = 'cuda' if self.args.use_cuda else 'cpu'
        return {
            'input_ids': torch.tensor([datapoint['input_ids'] for datapoint in batch]).long().to(device),
            'attention_mask': torch.tensor([datapoint['attention_mask'] for datapoint in batch]).long().to(device),
            'labels': torch.tensor([datapoint['label'] for datapoint in batch]).long().to(device),
            'rationales': torch.tensor([datapoint['rationale'] for datapoint in batch]).long().to(device),
        }
