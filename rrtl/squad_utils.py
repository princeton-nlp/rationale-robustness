import argparse
from collections import Counter, OrderedDict, defaultdict
import json
import re
import string
import sys

### BEGIN: official SQuAD code version 1.1
### See https://rajpurkar.github.io/SQuAD-explorer/
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    normalized_text = white_space_fix(remove_articles(remove_punc(lower(s))))
    return normalized_text


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
### END: official SQuAD code


# The functions below aims to achieve the same goal as: https://github.com/huggingface/transformers/blob/010965dcde8ce9526f6a7e6e2c3f36276c153708/src/transformers/data/metrics/squad_metrics.py#L384
def get_nbest_indicies(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_valid_best_span(batch_start_logits, batch_end_logits):
    """
    start_logits: batched tensor
    end_logits: batched tensor
    """
    n_best_size = 20
    max_answer_length = 30
    pred_start_positions = []
    pred_end_positions = []

    for i in range(batch_start_logits.size(0)):
        start_logits = batch_start_logits[i].tolist()
        end_logits = batch_end_logits[i].tolist()

        start_inds = get_nbest_indicies(start_logits, 20)
        end_inds = get_nbest_indicies(end_logits, 20)
        pred_spans_and_logits = []
        for start_ind in start_inds:
            for end_ind in end_inds:
                if end_ind < start_ind:
                    continue
                length = end_ind - start_ind + 1
                if length > max_answer_length:
                    continue
                pred_spans_and_logits.append(
                    ((start_ind, end_ind), start_logits[start_ind] + end_logits[end_ind])
                )
        pred_start_position, pred_end_position = sorted(pred_spans_and_logits, key=lambda x: -x[1])[0][0]
        pred_start_positions.append(pred_start_position)
        pred_end_positions.append(pred_end_position)
    return pred_start_positions, pred_end_positions






