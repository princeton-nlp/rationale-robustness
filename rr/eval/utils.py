import json
from argparse import Namespace

import numpy as np
from termcolor import colored
from IPython.display import clear_output

from rr.config import Config

config = Config()


def recover_to_pre_aug(sent_rationales, augs):
    """
    >>> sent_rationales = [0, 2, 3, 5, 8, 9, 10, 13, 18]  # augmented sequence
    >>> augs = [7, 12, 20]
    >>> recover_to_pre_aug(sent_rationales, augs)
    [0, 2, 3, 5, 7, 8, 9, 11, 16]
    """
    augs = sorted(augs, reverse=True)
    recovered_sent_rationales = []
    for aug in augs:
        for rationale in sent_rationales:
            if rationale < aug:
                recovered_sent_rationales.append(rationale)
            elif rationale > aug:
                recovered_sent_rationales.append(rationale - 1)
            else:
                # this could happen when the `sent_rationales` is predicted instead of gold
                continue
        sent_rationales, recovered_sent_rationales = recovered_sent_rationales, []
    return sent_rationales


def get_sent_token_ranges(sentences):
    sent_lengths = [len(s.strip().split()) for s in sentences]
    accum_lengths = [0] + np.cumsum(sent_lengths).tolist()
    sent_token_ranges = [(accum_lengths[i], accum_lengths[i + 1]) for i in range(len(accum_lengths) - 1)]
    return sent_token_ranges


def get_sent_level_rationale_from_token_ranges(rationale_token_ranges, sent_token_ranges):
    sent_rationales = []
    i = 0
    j = 0
    while i < len(sent_token_ranges) and j < len(rationale_token_ranges):
#        print(f'sent range (i={i}): {sent_token_ranges[i]} | rationale range (j={j}): {rationale_token_ranges[j]}')
#        print(f'prev sent range (i={i - 1}): {sent_token_ranges[i - 1]} | prev rationale range (j={j - 1}): {rationale_token_ranges[j - 1]}')
#        print(sent_rationales)
#        print('---\n')
        start_sent, end_sent = sent_token_ranges[i]
        start_ra, end_ra = rationale_token_ranges[j]

        if start_sent <= start_ra and end_sent >= end_ra:
            sent_rationales.append(i)
            j += 1  # matched a rationale, move on to the next
        elif start_ra >= end_sent:
            i += 1
        elif start_sent <= start_ra and end_sent < end_ra:
            sent_rationales.append(i)
            i += 1
        elif start_sent > start_ra and end_sent >= end_ra:
            sent_rationales.append(i)
            j += 1
        elif start_sent > start_ra and end_sent < end_ra:
            sent_rationales.append(i)
            i += 1

    return list(set(sent_rationales))


def get_annotation_id_and_doc_handle(dataset_name, path_type, obj, doc_dir=None):
    """
    path_type (str): either `pred_path` or `dev_path`
    """
    if dataset_name == 'movies':
        if path_type == 'pred_path':
            docf = open(doc_dir + obj.annotation_id.split(':')[0])
            annotation_id = obj.annotation_id.split(':')[0]
        else:
            docf = None
            annotation_id = obj.annotation_id

    elif dataset_name in ('fever', 'multirc'):
        if path_type == 'pred_path':
            docf = open(doc_dir + obj.rationales[0]['docid'])
            annotation_id = obj.annotation_id
        else:
            docf = None
            annotation_id = obj.annotation_id

    return annotation_id, docf


def load_gold_and_pred_rationales(args, pred_path, dev_path, doc_dir, debug=False, show_incorrect_only=False, specified_annotation_id=None):
    target_id_to_name = {v: k for k, v in config.target_vocab[args.dataset_name].items()}

    id_to_sentences = {}
    id_to_sent_token_ranges = {}
    id_to_pred_rationales = {}
    id_to_pred_sent_rationales = {}
    id_to_gold_target = {}
    id_to_pred_target = {}
    id_to_query = {}
    with open(pred_path) as f:
        for line in f:
            obj = json.loads(line)
            obj = Namespace(**obj)

            annotation_id, docf = get_annotation_id_and_doc_handle(
                args.dataset_name,
                path_type='pred_path',
                obj=obj,
                doc_dir=doc_dir
            )
            sentences = docf.readlines()
            id_to_sentences[annotation_id] = sentences

            pred_rationales = sorted([
                (r['start_token'], r['end_token'])
                for r in obj.rationales[0]['hard_rationale_predictions']
            ])
            id_to_pred_rationales[annotation_id] = pred_rationales

            sent_token_ranges = get_sent_token_ranges(sentences)
            pred_sent_rationales = get_sent_level_rationale_from_token_ranges(pred_rationales, sent_token_ranges)
            id_to_pred_sent_rationales[annotation_id] = pred_sent_rationales

            id_to_gold_target[annotation_id] = obj.gold_classification
            id_to_pred_target[annotation_id] = target_id_to_name[int(obj.classification)]
            id_to_query[annotation_id] = obj.query

    id_to_gold_rationales = {}
    id_to_gold_sent_rationales = {}
    id_to_attack_sent_positions = {}
    with open(dev_path) as f:
        for line in f:
            obj = json.loads(line)
            obj = Namespace(**obj)

            annotation_id, _ = get_annotation_id_and_doc_handle(
                args.dataset_name,
                path_type='dev_path',
                obj=obj,
                doc_dir=None
            )

            gold_rationales = sorted([(r[0]['start_token'], r[0]['end_token']) for r in obj.evidences])
            id_to_gold_rationales[annotation_id] = gold_rationales
            sentences = id_to_sentences[annotation_id]
            sent_token_ranges = get_sent_token_ranges(sentences)
            gold_sent_rationales = get_sent_level_rationale_from_token_ranges(gold_rationales, sent_token_ranges)
            id_to_gold_sent_rationales[annotation_id] = gold_sent_rationales
#            print(annotation_id)
#            print(list(enumerate(sent_token_ranges)))
#            print(gold_rationales)
#            print(gold_sent_rationales)
#            input()
            id_to_attack_sent_positions[annotation_id] = obj.augs if hasattr(obj, 'augs') else [-1]

    id_to_pred_gold_rationales = {}
    annotation_ids = sorted(id_to_sentences.keys())
    for annotation_id in annotation_ids:
        if specified_annotation_id is not None and specified_annotation_id != annotation_id:
            continue
        sentences = id_to_sentences[annotation_id][:args.max_num_sentences]
        all_tokens = [tok for sent in sentences for tok in sent.strip().split()][:args.max_seq_length]
        pred_rationales = [(s, e) for s, e in id_to_pred_rationales[annotation_id] if e <= args.max_seq_length]
        gold_rationales = [(s, e) for s, e in id_to_gold_rationales[annotation_id] if e <= args.max_seq_length]
        pred_sent_rationales = [pos for pos in id_to_pred_sent_rationales[annotation_id] if pos < args.max_num_sentences]
        gold_sent_rationales = [pos for pos in id_to_gold_sent_rationales[annotation_id] if pos < args.max_num_sentences]
        augs = id_to_attack_sent_positions[annotation_id]
        pred_target = id_to_pred_target[annotation_id]
        gold_target = id_to_gold_target[annotation_id]
        

        # NOTE: The ERASER annotation contains rationales where `end_sentence` - `start_sentence` > 1.
        # NOTE: This might cause the augs and gold_sent_rationales to overlap when
        # NOTE: the insert position is between `end_sentence` and `start_sentence`.
        # NOTE: Here we work around it by moving the gold overlapping position by one index to the right
        # NOTE: **The token level issue is not solved yet. The fix is on sentence level.**
        aug_set = set(augs)
        gold_sent_rationales = [pos if pos not in aug_set else pos + 1 for pos in gold_sent_rationales]
        if (set(gold_sent_rationales) & aug_set):
            pass
            #print(annotation_id)
            #print(gold_sent_rationales)
            #print(augs)
            #input()

        if show_incorrect_only and gold_target == pred_target:
            continue

        if debug:
#            colored_tokens = color(all_tokens, pred_rationales, gold_rationales)
#            print(' '.join(colored_tokens))
#            print('^^^ Token level highlight.\n\n\n')
#            clear_output(wait=True)
            print('annotation_id:', annotation_id)
            print('gold rationales:', gold_sent_rationales)
            print('pred rationales:', pred_sent_rationales)
            print('augs:', augs)
            print('gold:', gold_target)
            print('pred:', pred_target)
            print()
            print('query:', id_to_query[annotation_id])
            print('---')
            print(''.join(color_sent_level(sentences, pred_sent_rationales, gold_sent_rationales, augs)))
            input()

        id_to_pred_gold_rationales[annotation_id] = {
            'sentences': sentences,
            'all_tokens': all_tokens,
            'pred_rationales': pred_rationales,
            'pred_sent_rationales': pred_sent_rationales,
            'gold_rationales': gold_rationales,
            'gold_sent_rationales': gold_sent_rationales,
            'augs': augs,
            'pred_target': pred_target,
            'gold_target': gold_target
        }
    return id_to_pred_gold_rationales


def color_sent_level(sentences, pred_pos, gold_pos, augs=None):
    colored_sentences = []
    pred_pos = set(pred_pos)
    gold_pos = set(gold_pos)
    if augs is None:
        aug = set()

    for i, sentence in enumerate(sentences):
        if i in pred_pos and i in gold_pos:
            sentence = colored(sentence, 'green')
        elif i in pred_pos:
            sentence = colored(sentence, 'red')
        elif i in gold_pos:
            sentence = colored(sentence, 'blue')
        if i in augs:
            sentence = colored('<<Attack>> ', 'yellow') + sentence
            
        colored_sentences.append(f'[{i}]: ' + sentence)
    return colored_sentences
