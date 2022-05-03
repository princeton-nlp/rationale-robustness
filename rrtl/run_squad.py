"""
Main training script for SQuAD.
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
from rrtl.logging_utils import log_qa
from rrtl.squad_utils import (
    metric_max_over_ground_truths,
    exact_match_score,
    f1_score,
    get_valid_best_span,
)
from rrtl.config import Config

config = Config()


def train_epoch(epoch, model, dl, optimizer, args):
    model.train()
    num_batches = dl.train.dataset.num_batches
    args.epoch = epoch

    t = time()
    total_loss = []
    total_kl_loss = []
    total_sent_rationale_loss = []
    total_neg_rationale_loss = []
    total_gold_sent_acc_thres05 = []
    total_pred_loss = []
    total_em_scores = []
    total_f1_scores = []

    for batch_idx, batch in enumerate(dl.train):
        output = model(batch)
        if args.inspect_gpu:
            os.system('nvidia-smi')
            input()
        loss = output['loss'].mean()
        pred_start_logits = output['start_logits']
        pred_end_logits = output['end_logits']
        input_texts = batch['input_texts']
        token_offsets = batch['token_offsets']
        pred_start_positions = torch.argmax(pred_start_logits, dim=1)
        pred_end_positions = torch.argmax(pred_end_logits, dim=1)

        pred_start_positions = pred_start_positions.tolist()
        pred_end_positions = pred_end_positions.tolist()
        pred_answers = [input_text[o[s][0]:o[e - 1][1]] for s, e, o, input_text in zip(pred_start_positions, pred_end_positions, token_offsets, input_texts)]
        gold_answers = batch['gold_answers']
        
        em_scores = [metric_max_over_ground_truths(exact_match_score, pred, golds) for pred, golds in zip(pred_answers, gold_answers)]
        f1_scores = [metric_max_over_ground_truths(f1_score, pred, golds) for pred, golds in zip(pred_answers, gold_answers)]
        total_em_scores += em_scores
        total_f1_scores += f1_scores

        total_loss.append(loss.item())
        total_kl_loss.append(output['kl_loss'].mean().item() if 'kl_loss' in output else 0.0)
        total_pred_loss.append(output['pred_loss'].mean().item() if 'pred_loss' in output else 0.0)
        total_sent_rationale_loss.append(output['sent_rationale_loss'].mean().item() if 'sent_rationale_loss' in output else 0.0)
        total_neg_rationale_loss.append(output['neg_rationale_loss'].mean().item() if 'neg_rationale_loss' in output else 0.0)
        total_gold_sent_acc_thres05.append(
            (output['gold_sent_probs'] > 0.5).float().mean().item()
            if 'gold_sent_probs' in output else 0.0
        )
        args.total_seen += pred_start_logits.size(0)

        if args.grad_accumulation_steps > 1:
            loss = loss / args.grad_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        if (args.global_step + 1) % args.grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if (args.global_step + 1) % args.print_every == 0:
                train_loss = sum(total_loss) / len(total_loss)
                train_em = sum(total_em_scores) / len(total_em_scores)
                train_f1 = sum(total_f1_scores) / len(total_f1_scores)
                train_kl_loss = sum(total_kl_loss) / len(total_kl_loss)
                train_pred_loss = sum(total_pred_loss) / len(total_pred_loss)
                train_sent_rationale_loss = sum(total_sent_rationale_loss) / len(total_sent_rationale_loss)
                train_neg_rationale_loss = sum(total_neg_rationale_loss) / len(total_neg_rationale_loss)
                train_gold_sent_acc_thres05 = sum(total_gold_sent_acc_thres05) / len(total_gold_sent_acc_thres05)
    
                elapsed = time() - t
                log_dict = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'num_batches': num_batches,
                    'loss': train_loss,
                    'train_kl_loss': train_kl_loss,
                    'train_pred_loss': train_pred_loss,
                    'train_sent_rationale_loss': train_sent_rationale_loss,
                    'train_neg_rationale_loss': train_neg_rationale_loss,
                    'train_gold_sent_acc_thres05': train_gold_sent_acc_thres05,
                    'train_em': train_em,
                    'train_f1': train_f1,
                    'elapsed': elapsed,
                    'global_step': args.global_step,
                }
                log_qa('train', args, log_dict)
                if args.wandb:
                    wandb.log(log_dict)
                total_loss = []
                total_kl_loss = []
                total_pred_loss = []
                total_sent_rationale_loss = []
                total_neg_rationale_loss = []
                total_gold_sent_acc_thres05 = []
                total_em_scores = []
                total_f1_scores = []
    
                t = time()
            
            if (args.global_step + 1) % args.eval_interval == 0:
                dev_em, dev_f1 = evaluate('dev', model, dl, args)
                if dev_f1 > args.best_score:
                    args.best_score = dev_f1
                    args.dev_f1 = dev_f1
                    args.dev_em = dev_em
                    if args.ckpt:
                        save_ckpt(args, model, optimizer, latest=False)
                if args.ckpt:
                    save_ckpt(args, model, optimizer, latest=True)
    
                model.train()
        args.global_step += 1


def evaluate(mode, model, dl, args):
    model.eval()

    total_loss = []
    total_em_scores = []
    total_f1_scores = []
    for batch_idx, batch in enumerate(dl[mode]):
        output = model.module.forward_eval(batch) if args.dataparallel else  model.forward_eval(batch) 
        loss = output['loss'].tolist()
        pred_start_logits = output['start_logits']
        pred_end_logits = output['end_logits']
#        pred_start_positions = torch.argmax(pred_start_logits, dim=1).tolist()
#        pred_end_positions = torch.argmax(pred_end_logits, dim=1).tolist()
        pred_start_positions, pred_end_positions = get_valid_best_span(pred_start_logits, pred_end_logits)


        token_offsets = batch['token_offsets']
        input_texts = batch['input_texts']
        gold_answers = batch['gold_answers']

        # mapping the predicted token positions back to char-level span offsets
        # note that the trailing token position is skipped because the previous char span will cover it
        pred_answers = [input_text[o[s][0]:o[e - 1][1]] for s, e, o, input_text in zip(pred_start_positions, pred_end_positions, token_offsets, input_texts)]
        em_scores = [metric_max_over_ground_truths(exact_match_score, pred, golds) for pred, golds in zip(pred_answers, gold_answers)]
        f1_scores = [metric_max_over_ground_truths(f1_score, pred, golds) for pred, golds in zip(pred_answers, gold_answers)]

        total_em_scores += em_scores
        total_f1_scores += f1_scores
        total_loss += [loss] if isinstance(loss, float) else loss
    loss = sum(total_loss) / len(total_loss)
    eval_em = sum(total_em_scores) / len(total_em_scores)
    eval_f1 = sum(total_f1_scores) / len(total_f1_scores)
    log_dict = {
        'epoch': args.epoch, 'batch_idx': batch_idx, 'global_step': args.global_step,
        'eval_em': eval_em, 'eval_f1': eval_f1, 'eval_loss': loss,
    }
    log_qa(mode, args, log_dict)
    if args.wandb:
        wandb.log(log_dict)
    return eval_em, eval_f1


def main(args):
    training_start_time = time()
    args = args_factory(args)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    save_args(args)
    if args.wandb:
        wandb.init(project=config.PROJECT_NAME, name=args.run_name)

    dataloader_class = get_dataloader_class(args)
    dl = dataloader_class(args)

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
    print(f'Best dev result: EM = {args.dev_em * 100:.2f}, F1 = {args.dev_f1 * 100:.2f}')
    print(f'Full run time elapsed: {(time() - training_start_time) / 60:.2f} min')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True, help="[squad]")
    parser.add_argument("--dataset-split", type=str, default="all", help="[all | train | dev | test]")
    parser.add_argument("--model-type", type=str, required=True, help="[fc_squad | vib_squad_sent]")
    parser.add_argument("--encoder-type", type=str, default="bert-base-uncased")
    parser.add_argument("--decoder-type", type=str, default="bert-base-uncased")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=config.CACHE_DIR)
    parser.add_argument("--overwrite_cache", default=True)

    # cuda
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--inspect-gpu", action="store_true")
    parser.add_argument("--disable-cuda", action="store_true")

    # printing, logging, and checkpointing
    parser.add_argument("--print-every", type=int, default=80)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--disable-ckpt", action="store_true")

    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=400)
    parser.add_argument("--num_epoch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)

    # VIB model
    parser.add_argument("--pi", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gamma2", type=float, default=1.0)
    parser.add_argument("--use-gold-rationale", action="store_true")
    parser.add_argument("--use-neg-rationale", action="store_true")

    # VIB sentence-level model
    parser.add_argument("--mask-scoring-func", type=str, default="linear", help="[linear | query]")
    parser.add_argument("--flexible-prior", type=str, default=None, help="[small-gold-fixed | mid-gold-fixed | large-gold-fixed]")
    parser.add_argument("--flexible-gold", nargs='+', type=float, default=[0.0, 1.0], 
        help="How much the gold rationale should be weighted. Value \in [0, 1]")

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

    main(args)