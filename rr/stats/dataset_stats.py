"""
Calculate dataset stats
"""
import os
import json
import argparse
from collections import Counter

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