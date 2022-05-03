import argparse

import torch.nn as nn

from rrtl.run_squad import evaluate
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
    evaluate(ckpt_args.dataset_split, model, dl, ckpt_args)


if __name__ == '__main__':
    """
    Eval on SQuAD dev set:
        python -m rrtl.run_squad_eval --dataset-name squad --load-path /path/to/your/checkpoint/
    Eval on adv-SQuAD addsent:
        python -m rrtl.run_squad_eval --dataset-name squad-addonesent --load-path /path/to/your/checkpoint/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset to evaluate on. [squad]")
    parser.add_argument("--dataset-split", type=str, default="dev", help="[train | dev | test]")
    parser.add_argument("--eval-mode", type=str, default="eval", help="[eval | stats | vis]")
    parser.add_argument("--eval_pi", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_eval(args)
