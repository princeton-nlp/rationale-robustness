"""
Find all `rationale_predictions.json` files under the given directory.
"""
import os
import json
import argparse
from argparse import Namespace
from collections import defaultdict

from rr.config import Config

config = Config()


def get_metric_from_file(args, path):
    target_id_to_name = {v: k for k, v in config.target_vocab[args.dataset_name].items()}
    correct = 0
    total = 0
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            obj = Namespace(**obj)
            if obj.gold_classification == target_id_to_name[int(obj.classification)]:
                correct += 1
            total += 1
    print(correct)
    print(total)
    print(correct / total)
    print('---')
    return correct / total


def collect_beam_search_pred_paths(args, dir_path):
    results = []
    for name in os.listdir(dir_path):
        model_dir_path = os.path.join(dir_path, name)
        if not os.path.isdir(model_dir_path):
            continue

        for dir_ in os.listdir(model_dir_path):
            path = os.path.join(model_dir_path, dir_, 'rationale_predictions.json')
            if os.path.exists(path):
                print(model_dir_path)
                metric = get_metric_from_file(args, path)
                results.append((name, dir_, metric))
    return results


def plot():
    pass


def main(args):
    dir_path = os.path.join('/n/fs/nlp-hc22/rationale-robustness/predictions', args.pred_dir, args.bottleneck_type)
    results = collect_beam_search_pred_paths(args, dir_path)
    results = sorted(results, key=lambda x: (x[0], x[1]))

    pred_strings = []
    if args.agg_attacks:
        agg_results = defaultdict(list)
        for model_name, dir_, score in results:
            if dir_ == 'original':
                agg_results[(model_name, 'original')] = [score]
            else:
                agg_results[(model_name, 'attacked')].append(score)
        for (model_name, dir_), scores in agg_results.items():
            if dir_ == 'attacked':
                orig_score = agg_results[(model_name, 'original')][0]
                avg_attack_score = sum(scores) / len(scores)
                diff = orig_score - avg_attack_score
                pred_strings.append(f'{model_name} | {dir_} | {sum(scores) / len(scores) * 100:.2f} | diff={diff * 100:.2f}')
            else:
                pred_strings.append(f'{model_name} | {dir_} | {sum(scores) / len(scores) * 100:.2f}')
    else:
        for model_name, dir_, score in results:
            pred_strings.append(f'{model_name} | {dir_} | {sum(scores) / len(scores) * 100:.2f}')
    
#    with open(os.path.join(dir_path, 'predictions.txt'), 'w') as f:
#        for pred_string in pred_strings:
#            print(pred_string)
#            f.write(pred_string + '\n')
    for pred_string in pred_strings:
        print(pred_string)
    print(f'Total: {len(results)}')


if __name__ == '__main__':
    """
    python -m rr.stats.read_beam_search_stats --dataset-name fever --bottleneck-type vib --pred-dir fever_beam_search --agg-attacks

    dir should be structured as: top_dir/bottleneck_type/model_name/orig_or_attack_dirs/pred.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True, help="[fever | multirc | imdb]")
    parser.add_argument("--bottleneck-type", type=str, required=True, help="[vib | vib_semi | full | full_multitask]")
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument("--agg-attacks", action="store_true")
    args = parser.parse_args()
    main(args)