import argparse

import torch.nn as nn

from rrtl.run_eraser import (
    evaluate,
    evaluate_for_stats,
    evaluate_for_vis,
)
from rrtl.utils import (
    load_ckpt,
    update_args,
    get_dataloader_class,
)


def eval_args_factory(ckpt_args, args):
    """
    Adjust args temporarily for running evaluation.
    """
    ckpt_args = update_args(ckpt_args)
    ckpt_args.no_shuffle = True
    ckpt_args.dataparallel = False
    ckpt_args.batch_size = 1
    ckpt_args.global_step = -1
    ckpt_args.epoch = -1
    ckpt_args.wandb = False
    ckpt_args.use_gold_rationale = False
    ckpt_args.use_neg_rationale = False
    ckpt_args.dataset_name = args.dataset_name
    ckpt_args.dataset_split = args.dataset_split
    ckpt_args.debug = args.debug
    ckpt_args.attack_path = args.attack_path
    return ckpt_args


def load_for_eval(args):
    model, ckpt_args, _ = load_ckpt(args.load_path)
    ckpt_args = eval_args_factory(ckpt_args, args)
    model = model.cuda()
    dataloader_class = get_dataloader_class(ckpt_args)
    dl = dataloader_class(ckpt_args)
    return model, dl, ckpt_args


def run_eval(args):
    model, dl, ckpt_args = load_for_eval(args)
    if args.eval_mode == 'eval':
        evaluate(ckpt_args.dataset_split, model, dl, ckpt_args)
    elif args.eval_mode == 'stats':
        macro_f1, micro_f1 = evaluate_for_stats(ckpt_args.dataset_split, model, dl, ckpt_args)
        print(f'Gold Rationale Capture Rate | macro f1: {macro_f1 * 100:.2f}')
        print(f'Gold Rationale Capture Rate | micro f1: {micro_f1 * 100:.2f}')
    elif args.eval_mode == 'vis':
        evaluate_for_vis(ckpt_args.dataset_split, model, dl, ckpt_args)


if __name__ == '__main__':
    """
    python -m rrtl.run_eraser_eval --dataset-name beer --load-path /path/to/your/checkpoint/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset to evaluate on. [beer | hotel | ...]")
    parser.add_argument("--dataset-split", type=str, default="dev", help="[train | dev | test]")
    parser.add_argument("--eval-mode", type=str, default="eval", help="[eval | stats | vis]")
    parser.add_argument("--eval_pi", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_eval(args)
