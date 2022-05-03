"""
Main training script for Beer and Hotel.
"""
import os
import sys
import argparse
from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

from rrtl.utils import (
    get_model_class,
    get_optimizer_class,
    get_dataloader_class,
    args_factory,
    save_args,
    save_ckpt,
    load_ckpt,
)
from rrtl.logging_utils import log, visualize_rationale
from rrtl.stats import gold_rationale_capture_rate
from rrtl.config import Config

config = Config()


def train_epoch(epoch, model, dl, optimizer, args):
    model.train()
    num_batches = dl.train.dataset.num_batches
    args.epoch = epoch

    t = time()
    total_loss = 0
    total_kl_loss = 0  # does not apply to non-VIB models
    total_correct = []

    for batch_idx, batch in enumerate(dl.train):
        output = model(batch)
        loss = output['loss']
        logits = output['logits']
        if args.inspect_gpu:
            os.system('nvidia-smi')
            input()

        if args.dataparallel:
            if 'kl_loss' in output:
                total_kl_loss += output['kl_loss'].sum().item()
            total_loss += loss.sum().item()
            total_correct += (torch.argmax(logits, dim=1) == batch['labels']).tolist()
            args.total_seen += logits.size(0)
            loss = loss.mean()
        else:
            if 'kl_loss' in output:
                total_kl_loss += output['kl_loss'].item()
            total_loss += loss.item()
            total_correct += (torch.argmax(logits, dim=1) == batch['labels']).tolist()
            args.total_seen += logits.size(0)

        if args.grad_accumulation_steps > 1:
            loss = loss / args.grad_accumulation_steps
        
        loss.backward()

        if (args.global_step + 1) % args.grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            if (args.global_step + 1) % args.print_every == 0:
                train_acc = sum(total_correct) / len(total_correct)
                train_loss = total_loss / len(total_correct)
                kl_loss = total_kl_loss / len(total_correct)
    
                elapsed = time() - t
                log_dict = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'num_batches': num_batches,
                    'acc': train_acc,
                    'loss': train_loss,
                    'elapsed': elapsed,
                    'global_step': args.global_step,
                    'kl_loss': kl_loss,
                }
                log('train', args, log_dict)
                if args.wandb:
                    wandb.log(log_dict)
                total_loss = 0
                total_kl_loss = 0
                total_correct = []
                t = time()
            
            if (args.global_step + 1) % args.eval_interval == 0:
                dev_acc = evaluate('dev', model, dl, args)
                if dev_acc > args.best_score:
                    args.best_score = dev_acc
                    if args.ckpt:
                        save_ckpt(args, model, optimizer, latest=False)
                if args.ckpt:
                    save_ckpt(args, model, optimizer, latest=True)
    
                model.train()
        args.global_step += 1


def evaluate(mode, model, dl, args):
    model.eval()

    total_loss = 0
    total_correct = []
    for batch_idx, batch in enumerate(dl[mode]):
        if args.dataparallel:
            output = model.module.forward_eval(batch)
        else:
            output = model.forward_eval(batch)
        loss = output['loss']
        logits = output['logits']

        if args.dataparallel:
            total_loss += loss.sum().item()
        else:
            total_loss += loss.item()
        total_correct += (torch.argmax(logits, dim=1) == batch['labels']).tolist()
    acc = sum(total_correct) / len(total_correct)
    loss = total_loss / len(total_correct)
    log_dict = {
        'epoch': args.epoch,
        'batch_idx': batch_idx,
        'eval_acc': acc,
        'eval_loss': loss,
        'global_step': args.global_step,
    }
    log(mode, args, log_dict)
    if args.wandb:
        wandb.log(log_dict)
    return acc


def evaluate_for_stats(mode, model, dl, args):
    model.eval()
    tps = []
    fps = []
    fns = []
    for batch_idx, batch in enumerate(dl[mode]):
        if args.dataparallel:
            output = model.module.forward_eval(batch)
        else:
            output = model.forward_eval(batch)
        loss = output['loss']
        logits = output['logits']
        batch_tps, batch_fps, batch_fns = gold_rationale_capture_rate(args, batch, output, dl.tokenizer)
        tps += batch_tps
        fps += batch_fps
        fns += batch_fns

    # calc macro f1
    f1s = [2 * tp / (2 * tp + fp + fn) for tp, fp, fn in zip(tps, fps, fns)]
    macro_f1 = sum(f1s) / len(f1s)
    
    # calc micro f1
    micro_f1 = 2 * sum(tps) / (2 * sum(tps) + sum(fps) + sum(fns))
    return macro_f1, micro_f1


def evaluate_for_vis(mode, model, dl, args):
    model.eval()
    for batch_idx, batch in enumerate(dl[mode]):
        if args.dataparallel:
            output = model.module.forward_eval(batch)
        else:
            output = model.forward_eval(batch)
        loss = output['loss']
        logits = output['logits']
        visualize_rationale(args, batch, output, dl.tokenizer)


def main(args):
    training_start_time = time()
    args = args_factory(args)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    save_args(args)
    if args.wandb:
        wandb.init(project=config.PROJECT_NAME, name=args.run_name)

    ## save examples ## 
    cached_features_file = os.path.join(args.cache_dir, 'cached_{}_{}_ml{}_bz{}'.format(
        args.dataset_name,
        args.dataset_split,
        str(args.max_length),
        args.batch_size,
    ))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("Loading features from cached file:", cached_features_file)
        dl = torch.load(cached_features_file)
    else:
        dataloader_class = get_dataloader_class(args)
        dl = dataloader_class(args)
        print("Saving features into cached file:", cached_features_file)
        torch.save(dl, cached_features_file)

    model_class = get_model_class(args)
    model = model_class(args=args)
    model = model.cuda()
    if args.dataparallel:
        model = nn.DataParallel(model)

    print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'All params      : {sum(p.numel() for p in model.parameters())}')
    optimizer_class = get_optimizer_class(args)
    optimizer = optimizer_class(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epoch):
        train_epoch(epoch, model, dl, optimizer, args)

    save_args(args)
    print(f'Full run time elapsed: {(time() - training_start_time) / 60:.2f} min')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True, help="[fever | multirc]")
    parser.add_argument("--dataset-split", type=str, default="all", help="[all | train | dev | test]")
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--encoder-type", type=str, default="bert-base-uncased")
    parser.add_argument("--decoder-type", type=str, default="bert-base-uncased")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=config.CACHE_DIR)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--attack_path", type=str, default=None)

    # cuda
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--inspect-gpu", action="store_true")
    parser.add_argument("--disable-cuda", action="store_true")

    # printing, logging, and checkpointing
    parser.add_argument("--print-every", type=int, default=80)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--disable-ckpt", action="store_true")

    # training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)

    # VIB model
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--pi", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gamma2", type=float, default=1.0)
    parser.add_argument("--use-gold-rationale", action="store_true")
    parser.add_argument("--use-neg-rationale", action="store_true")
    parser.add_argument("--fix-input", type=str, default=None)
    
    # SPECTRA model
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--budget_ratio", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--solver_iter", type=int, default=100)

    args = parser.parse_args()

    if args.debug:
        args.print_every = 2
        args.batch_size = 3
        args.eval_interval = 20
        args.num_epoch = 100
        args.dataparallel = False
        args.overwrite_cache = False
        args.max_length = 200

    main(args)