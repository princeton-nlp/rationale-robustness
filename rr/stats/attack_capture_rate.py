"""
TODO: clean up and move functions to common utils
TODO: manage paths
"""
import json
import argparse
import numpy as np


def get_full_sent_rationales(path, sparsity):
    sent_rationales = []
    with open(path) as f:
        for line in f:
            docid, probs, sent_mask = line.strip().split('<SEP>')
            probs = json.loads(probs.strip())
            sent_mask = json.loads(sent_mask.strip())
            #rationales = [1 if p > 0.5 else 0 for p in probs]
            num_rationales = int(sum(sent_mask) * sparsity)
            topk_probs_inds = sorted(range(len(probs)), key=lambda i: probs[i])[-num_rationales:]
            topk_probs_inds = set(topk_probs_inds)
            rationales = [1 if i in topk_probs_inds else 0 for i in range(len(probs))]
            sent_rationales.append(rationales)
    return sent_rationales


def get_vib_sent_rationales(path):
    sent_rationales = []
    with open(path) as f:
        for line in f:
            docid, rationales, sent_mask = line.strip().split('<SEP>')
            rationales = json.loads(rationales)
            sent_mask = json.loads(sent_mask.strip())
            sent_rationales.append(rationales)
    return sent_rationales


def calc_attack_capture_rate(sent_rationales, attack_pos):
    total = len(sent_rationales)
    num_selected = 0
    for sent_rationale in sent_rationales:
        if sent_rationale[attack_pos] == 1:
            num_selected += 1
    return num_selected / total


def main(args):
    if args.dataset_name == 'fever':
        #attack_positions = list(range(0, 10))
        attack_positions = [0, 9]
        sparsity = 0.4 # fever
        #sparsity = 0.2 # fever
    elif args.dataset_name == 'multirc':
#        attack_positions = list(range(0, 14))
        attack_positions = [0, 9]
        sparsity = 0.25 # multirc
        #sparsity = 0.15 # multirc

    for pos in attack_positions:
        attack_dir = f'{args.attack_type}_pos{pos}'
#        rationale_path = args.rationale_path
#        rationale_path = f"/n/fs/nlp-hc22/rationale-robustness/predictions/{args.pred_dir}/{args.bottleneck_type}/{args.model_name}/{attack_dir}/sentence_probabilities.txt"
        rationale_path = f"/n/fs/nlp-hc22/rationale-robustness/experiments/{args.pred_dir}/{args.bottleneck_type}/{args.model_name}/sentence_probabilities.txt"
        #vib_sent_rationales = base + "vib/sentence_probabilities.txt"
        #vib_semi_sent_rationales = base + "vib_semi/sentence_probabilities.txt"
        #full_multitask_sent_rationales = base + "full_multitask/sentence_probabilities.txt"
        
        if args.bottleneck_type == 'full_multitask':
            rationales = get_full_sent_rationales(rationale_path, sparsity)
        elif args.bottleneck_type in ('vib', 'vib_semi'):
            rationales = get_vib_sent_rationales(rationale_path)
        #full_multitask_rationales = get_full_sent_rationales(full_multitask_sent_rationales, sparsity)
        #vib_rationales = get_vib_sent_rationales(vib_sent_rationales)
        #vib_semi_rationales = get_vib_sent_rationales(vib_semi_sent_rationales)
        #print('Prediction file base path:', base)
        print(f'pos={pos} | {args.bottleneck_type} | {calc_attack_capture_rate(rationales, attack_pos=pos) * 100:.2f}')
        #print(f'full_multitask: {calc_attack_capture_rate(full_multitask_rationales, attack_pos=pos) * 100:.2f}')
        #print(f'vib: {calc_attack_capture_rate(vib_rationales, attack_pos=pos) * 100:.2f}')
        #print(f'vib_semi: {calc_attack_capture_rate(vib_semi_rationales, attack_pos=pos) * 100:.2f}')


if __name__ == '__main__':
    """
    python -m rr.stats.attack_capture_rate --dataset-name fever --pred-dir fever_shuffle_sent_model --bottleneck-type vib
    python -m rr.stats.attack_capture_rate --dataset-name fever --bottleneck-type vib --pred-dir fever_beam_search --model-name fever_vib_pi\=0.4_beta\=2.0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True, help="[fever | multirc | imdb]")
    parser.add_argument("--bottleneck-type", type=str, required=True, help="[vib | vib_semi | full | full_multitask]")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--pred-dir", type=str, required=True, help="[fever_pi0.4 | fever_shuffle_sent_model | multirc_shuffle_sent_model]")
    parser.add_argument("--attack-type", type=str, default='addsent', help="[addsent | addrand | addwiki]")
    args = parser.parse_args()
    main(args)
