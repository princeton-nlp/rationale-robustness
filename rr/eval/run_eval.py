import os
import json
import argparse
from argparse import Namespace

import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset
)

import transformers
from transformers import (
    BertConfig, BertForSequenceClassification, BertTokenizer,
    DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,
)

from rr.base.explainable_qa.information_bottleneck.bert_explainer import (
    DistilBertExplainer,
    DistilBertSentenceExplainer
)
from rr.base.explainable_qa.information_bottleneck.gated_explainer import (
    DistilBertGatedSentenceExplainer,
    DistilBertHardGatedSentenceExplainer
)
from rr.base.explainable_qa.baselines.baseline_models import BertForTaskCumEvidenceClassfication

from rr.base.explainable_qa.information_bottleneck.ib_train_sentence import load_and_cache_examples as vib_load_and_cache_examples
from rr.base.explainable_qa.information_bottleneck.ib_train_sentence import evaluate as vib_evaluate
from rr.base.explainable_qa.baselines.evidence_deploy import load_and_cache_examples as full_load_and_cache_examples
from rr.base.explainable_qa.baselines.evidence_deploy import evaluate as full_evaluate
from rr.base.explainable_qa.baselines.evidence_multitask import load_and_cache_examples as full_multitask_load_and_cache_examples
from rr.base.explainable_qa.baselines.evidence_multitask import evaluate as full_multitask_evaluate

from rr.eval.model_args import Args


MODEL_CLASSES = {
    'bert': (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizer
    ),
    'distilbert_gated_sent': (
        DistilBertConfig,
        DistilBertGatedSentenceExplainer,
        DistilBertTokenizer
    ),
    'distilbert_hard_gated_sent': (
        DistilBertConfig,
        DistilBertHardGatedSentenceExplainer,
        DistilBertTokenizer
    ),
    'multitask_bert': (
        BertConfig,
        BertForTaskCumEvidenceClassfication,
        BertTokenizer
    )
}

DATA_AND_EVAL_CLASSES = {
    'vib': (vib_load_and_cache_examples, vib_evaluate),
    'vib_semi': (vib_load_and_cache_examples, vib_evaluate),
    'full': (full_load_and_cache_examples, full_evaluate),
    'full_multitask': (full_multitask_load_and_cache_examples, full_multitask_evaluate)
}


def load_model_and_data_from_checkpoint(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    # exchange information between config and model parameters as required
    config.num_labels = len(args.classes)
    config.max_query_length = args.max_query_length

    if args.bottleneck_type in ('vib', 'vib_semi'):
        model = model_class.from_pretrained(
            model_class,
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config,
            model_params=vars(args)
        )
    elif args.bottleneck_type == 'full':
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config,
        )
    elif args.bottleneck_type == 'full_multitask':
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config,
        )
    model_state = torch.load(args.model_ckpt_path, map_location=torch.device('cpu'))
    
    model.load_state_dict(model_state)
    model.eval()
    model = model.cuda()

    # load data
    load_and_cache_examples_func, evaluate_func = DATA_AND_EVAL_CLASSES[args.bottleneck_type]

    dataset, examples, features = load_and_cache_examples_func(
        args, vars(args), tokenizer,
        evaluate=True, split='val', output_examples=True
    )
    return model, tokenizer, evaluate_func, dataset, examples, features


if __name__ == '__main__':
    """
    python -Wignore -m rr.eval.run_eval --model-name fever_vib_pi=0.4_beta=1.0 --output-to-tmp
    python -Wignore -m rr.eval.run_eval --model-name fever_vib_pi=0.4_beta=1.0 --bottleneck-type vib --exp-dir fever_beam_search --attack-dir addsent_pos0 --output-to-tmp
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="")
    parser.add_argument("--exp-dir", type=str, required=True, help="")
    parser.add_argument("--bottleneck-type", type=str, required=True)
    parser.add_argument("--attack-dir", type=str, default=None)
    parser.add_argument("--output-to-tmp", action="store_true", help="Save to tmp dir.")
    parser.add_argument("--load_old_path", type=str, default=None, help="specify full path and override")
    args = parser.parse_args()

    CKPT_DIR = '/path/to/your/model/checkpoint/dir/'
    ATTACK_DATA_DIR = '/path/to/your/attack/data/dir/'
    TMP_DIR = '/path/to/tmp/dir/'

    BASE_DIR = f'{CKPT_DIR}/{args.exp_dir}/{args.bottleneck_type}/{args.model_name}'
    ARGS_PATH = os.path.join(BASE_DIR, 'args.json')
    with open(ARGS_PATH) as f:
        ckpt_args = json.load(f)
        ckpt_args = Namespace(**ckpt_args)
    CKPT_PATH = os.path.join(BASE_DIR, 'best_model/pytorch_model.bin')
    ckpt_args.model_ckpt_path = CKPT_PATH
    ckpt_args.per_gpu_eval_batch_size = 1
    ckpt_args.n_gpu = 1
    ckpt_args.device = torch.device("cuda")
    if args.attack_dir is None:
        ckpt_args.full_pred_dir = os.path.join(
            ckpt_args.base_pred_dir, ckpt_args.pred_dir,
            ckpt_args.bottleneck_type, ckpt_args.model_name, 'original'
        )
    else:
        ckpt_args.full_pred_dir = os.path.join(
            ckpt_args.base_pred_dir, ckpt_args.pred_dir,
            ckpt_args.bottleneck_type, ckpt_args.model_name, args.attack_dir
        )
        ckpt_args.data_dir = f'{ATTACK_DATA_DIR}/{ckpt_args.dataset_name}/{args.attack_dir}'
    if args.output_to_tmp:
        ckpt_args.full_pred_dir = TMP_DIR
    print(
        f'Running eval...\n'
        f'dataset: {ckpt_args.dataset_name}\n'
        f'bottleneck_type: {ckpt_args.bottleneck_type}\n'
        f'model_name: {args.model_name}\n'
        f'attack_dir: {args.attack_dir}'
    )
    
    model, tokenizer, evaluate_func, dataset, examples, features = load_model_and_data_from_checkpoint(ckpt_args)
    result = evaluate_func(
        ckpt_args, vars(ckpt_args), model, tokenizer,
        prefix=0, output_examples=True, split='val',
        dataset_examples_features=(dataset, examples, features)
    )
    print(f'Accuracy: {result[1]:.4f}')
    print('=====')
