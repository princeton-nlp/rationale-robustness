"""
Analysis/visualization for ERASER datasets with token-level models.
"""
import os
import argparse
from termcolor import colored

import torch
import transformers

from rrtl.run_eraser_eval import load_for_eval
from rrtl.config import Config

transformers.logging.set_verbosity_error()
config = Config()
torch.set_printoptions(precision=7)


def highlight(input_tokens, token_z, attack_token_positions):
    attack_set = set(attack_token_positions) if attack_token_positions is not None else set()
    pred_set = set([i for i, z in enumerate(token_z) if z == 1])

    text = []
    for token_idx, token in enumerate(input_tokens):
        if token_idx in pred_set:
            text.append(colored(token, 'red'))
        elif token == '[PAD]':
            continue
        else:
            text.append(token)
    return ' '.join(text[1:-1])


def vis(args, model, dl):
    """
    This function assumes eval batch_size = 1.
    """
    for batch_idx, batch in enumerate(dl[args.dataset_split]):
        output = model.forward_eval(batch)
        logits = output['logits']
        pred_label = torch.argmax(logits, dim=1).item()
        pred_name = args.label_to_name[pred_label]

        input_ids = batch['input_ids'].tolist()[0]
        attention_mask = batch['attention_mask']
        gold_label = batch['labels'].item()
        gold_name = args.label_to_name[gold_label]
    
        input_tokens = [dl.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
        token_z = output['token_z'].tolist()[0]

        highlighted_context_sents = highlight(input_tokens,
                                              token_z,
                                              attack_token_positions=None)
        print('Id:', batch_idx)
        print('Text:')
        print(highlighted_context_sents)
        print('Pred label:', pred_name)
        print('Gold label:', gold_name)
        print('---')
        input()


def stats(args, model, dl):
    """
    Calculate gold capture rate & attack capture rate.
    This function assumes eval batch_size = 1.
    """
    c = 0
    gtps = []
    gfps = []
    gfns = []

    atps = []
    afps = []
    afns = []
    attack_capture_rate = []
    corrects = []

    for batch_idx, batch in enumerate(dl[args.dataset_split]):
        output = model.forward_eval(batch)
        logits = output['logits']

        pred_label = torch.argmax(logits, dim=1).item()
        pred_name = args.label_to_name[pred_label]
        gold_label = batch['labels'].item()
        gold_name = args.label_to_name[gold_label]
        correct = pred_name == gold_name
        corrects.append(correct)

        if args.has_rationale:
            gold = set(i for i, r in enumerate(batch['rationales'][0].tolist()) if r == 1)
            if args.attack_pos is not None:
                attack = set(i for i, r in enumerate(batch['rationales'][0].tolist()) if r == -1)
            else:
                attack = set()
            pred = set([i for i, z in enumerate(output['token_z'].tolist()[0]) if z > 0.8])

            gtp = len(gold & pred)
            gfp = len(pred) - gtp
            gfn = len(gold) - gtp
            gtps.append(gtp)
            gfps.append(gfp)
            gfns.append(gfn)

            atp = len(attack & pred)
            afp = len(pred) - atp
            afn = len(attack) - atp
            atps.append(atp)
            afps.append(afp)
            afns.append(afn)
            if len(attack) == 0:
                attack_capture_rate.append(0.0)
                c += 1
            else:
                attack_capture_rate.append(len(attack & pred) / len(attack))
    print(c)
    print('=================')
    print(f'Dataset: {args.dataset_name}')
    print(f'Split: {args.dataset_split}')
    print(f'Model: {args.load_path}')
    print(f'Attack Position: {args.attack_pos}')
    print(f'Acc: {sum(corrects) / len(corrects) * 100:.1f}')

    if args.has_rationale:
        # calc gold macro f1
        gf1s = [2 * tp / (2 * tp + fp + fn) for tp, fp, fn in zip(gtps, gfps, gfns)]
        g_macro_f1 = sum(gf1s) / len(gf1s)
        print(f'gold macro f1: {g_macro_f1 * 100:.2f}')
    
        # calc gold micro f1
        g_micro_f1 = 2 * sum(gtps) / (2 * sum(gtps) + sum(gfps) + sum(gfns))
        print(f'gold micro f1: {g_micro_f1 * 100:.2f}')

        # calc attack macro f1
        af1s = [2 * tp / (2 * tp + fp + fn) for tp, fp, fn in zip(atps, afps, afns)]
        a_macro_f1 = sum(af1s) / len(af1s)
        print(f'attack macro f1: {a_macro_f1 * 100:.2f}')
    
        # calc attack micro f1
        a_micro_f1 = 2 * sum(atps) / (2 * sum(atps) + sum(afps) + sum(afns))
        print(f'attack micro f1: {a_micro_f1 * 100:.2f}')

        print(f'ACR: {sum(attack_capture_rate) / len(attack_capture_rate) * 100:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", type=str, required=True)
    parser.add_argument("--no-rationale", action="store_true")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset to evaluate on.")
    parser.add_argument("--dataset-split", type=str, default="dev", help="[train | dev | test]")
    parser.add_argument("--eval-mode", type=str, default="stats", help="[stats | vis]")
    parser.add_argument("--attack_pos", type=int, default=None, help="attack position")
    parser.add_argument("--attack_type", type=str, default=None, help="sent | rand | wiki")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    args.load_path = config.EXP_DIR / args.load_path
    args.has_rationale = not args.no_rationale
    if args.dataset_name == 'fever':
        args.label_to_name = {label: name for name, label in config.FEVER_LABEL.items()}
    elif args.dataset_name == 'multirc':
        args.label_to_name = {label: name for name, label in config.MULTIRC_LABEL.items()}
    elif args.dataset_name == 'beer':
        args.label_to_name = {label: name for name, label in config.BEER_LABEL.items()}
    elif args.dataset_name == 'hotel':
        args.label_to_name = {label: name for name, label in config.HOTEL_LABEL.items()}
    else:
        raise ValueError(f'Dataset name: {args.dataset_name}')

    if args.attack_pos is None:
        args.attack_path = None
    elif args.dataset_name == 'beer':
        if args.attack_type is None:
            args.attack_path = config.DATA_DIR / f'sentiment/adv/beer/beer0.{args.dataset_split}.{args.attack_pos}'
        else:
            args.attack_path = config.DATA_DIR / f'sentiment/adv/beer/beer0.{args.dataset_split}.{args.attack_pos}.{args.attack_type}'
    elif args.dataset_name == 'hotel':
        if args.attack_type is None:
            args.attack_path = config.DATA_DIR / f'sentiment/adv/hotel//hotel_Cleanliness.{args.dataset_split}.{args.attack_pos}.csv'
        else:
            args.attack_path = config.DATA_DIR / f'sentiment/adv/hotel//hotel_Cleanliness.{args.dataset_split}.{args.attack_pos}.{args.attack_type}.csv'

    model, dl, ckpt_args = load_for_eval(args)
    if args.eval_mode == 'vis':
        vis(args, model, dl)
    elif args.eval_mode == 'stats':
        stats(args, model, dl)
