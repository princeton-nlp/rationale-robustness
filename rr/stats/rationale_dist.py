"""
Calculate rationale distribution from 
1. Gold rationale files
2. Prediction files
"""
import os
import json
import argparse
from collections import Counter
from collections import defaultdict

from rr.config import Config

config = Config()

def get_path(args):
    if args.shuffle_dir is None:
        if args.mode == 'train':
            data_path = config.DATA_DIR / args.dataset_name / 'train.jsonl'
        elif args.mode == 'dev':
            data_path = config.DATA_DIR / args.dataset_name / 'val.jsonl'
        elif args.mode == 'test':
            data_path = config.DATA_DIR / args.dataset_name / 'test.jsonl'
        docs_dir = config.DATA_DIR / args.dataset_name / 'docs'
    else:
        if args.mode == 'train':
            data_path = config.SHUFFLE_TRAIN_DATA_PATH.format(args.dataset_name, args.shuffle_dir)
        elif args.mode == 'dev':
            data_path = config.SHUFFLE_DEV_DATA_PATH.format(args.dataset_name, args.shuffle_dir)
        elif args.mode == 'test':
            data_path = config.SHUFFLE_TEST_DATA_PATH.format(args.dataset_name, args.shuffle_dir)
        docs_dir = config.SHUFFLE_DOCS_DIR.format(args.dataset_name, args.shuffle_dir)
    docids = os.listdir(docs_dir)
    return data_path, docs_dir, docids


def calc_gold_rationale_dist(args):
    data_path, docs_dir, docids = get_path(args)

    if args.show_doc_length_dist:
        num_sents = []
        docid_to_num_sents = {}
        for docid in docids:
            with open(os.path.join(docs_dir, docid)) as f:
                num = len([line for line in f])
                docid_to_num_sents[docid] = num
                num_sents.append(num)
        num_sents = Counter(num_sents)
        num_sents = sorted([(length, count) for length, count in dict(Counter(num_sents)).items()], key=lambda x: x[0])
        print('Document length (#sentences) distribution')
        for length, count in num_sents:
            print(f'Length: {length} | Count: {count}')


    gold_rationale_positions = []
    rationale_per_example = []
    avg_rationale_to_sent_per_example = []
    with open(data_path) as f:
        for line in f:
            obj = json.loads(line)
            docid = obj['evidences'][0][0]['docid']
            count = 0
            for evidence in obj['evidences']:
                for e in evidence:
                    start_sentence = e['start_sentence']
                    gold_rationale_positions.append(start_sentence)
                    count += 1
            rationale_per_example.append(count)
            avg_rationale_to_sent_per_example.append(count / docid_to_num_sents[docid])
            
    gold_rationale_positions = sorted([(length, count) for length, count in dict(Counter(gold_rationale_positions)).items()], key=lambda x: x[0])
    print('Gold rationale distribution')
    for length, count in gold_rationale_positions:
        print(f'Length: {length} | Count: {count}')
    
    print(len(rationale_per_example))
    print(sum(rationale_per_example) / len(rationale_per_example))
    print(sum(avg_rationale_to_sent_per_example) / len(avg_rationale_to_sent_per_example))


def main(args):
    if args.bottleneck_type is None:
        calc_gold_rationale_dist(args)
    else:
        # TODO: copy from notebooks/rationale-robustness/analysis/notebooks/inspect-rationale-distribution-by-length.ipynb
        pass
    

if __name__ == '__main__':
    """
    Run:
        python -m rr.stats.rationale_dist --dataset-name fever --mode train --shuffle-dir rand_first10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="fever", help="[fever | multirc]")
    parser.add_argument("--bottleneck-type", type=str, default=None,
                        help="[vib | vib_semi | full | full_multitask] (None means gold rationale)")
    parser.add_argument("--mode", type=str, default="train", help="[train | dev | test]")
    parser.add_argument("--shuffle-dir", type=str, default=None, help="[rand_first10 | ...]")
    parser.add_argument("--show-doc-length-dist", action="store_true",
                        help="Show number of sentences in the documents. This takes a while for document IO.")
    args = parser.parse_args()

    print(
        f'=======================\n'
        f'Running...\n'
        f'dataset: {args.dataset_name}\n'
        f'bottleneck_type: {args.bottleneck_type}\n'
        f'shuffle_dir: {args.shuffle_dir}\n'
        f'----'
    )
    main(args)