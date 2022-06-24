"""
Calculate gold rationale capture rate (micro & macro f1 scores).
TODO: currently reusing the same function from visualization. Need to speed up.
TODO: major clean up
"""
import argparse

from rr.eval.model_args import Args
from rr.eval.utils import load_gold_and_pred_rationales


if __name__ == '__main__':
    """
    Run:
        python -m rr.stats.gold_capture_rate --dataset-name fever --pred-dir fever_tune_kl/fever_vib_pi0.4_beta1.0 --bottleneck-type vib
        python -m rr.stats.gold_capture_rate --dataset-name fever --pred-dir fever_beam_search --bottleneck-type vib --model-name fever_vib_pi\=0.4_beta\=2.0 --attack-type addrand
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True, help="[fever | multirc]")
    parser.add_argument("--bottleneck-type", type=str, required=True, help="[vib | vib_semi | full | full_multitask]")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--pred-dir", type=str, default=None, help="[None | fever_pi0.4 | ...]. If `None` then use `dataset_name`.")
    parser.add_argument("--attack-type", type=str, default='addsent', help="addsent | addrand | addwiki")
    parser.add_argument("--run-original", action="store_true", help="run original eval instead of attacks")
    args = parser.parse_args()

    if args.pred_dir is None:
        args.pred_dir = args.dataset_name

    ckpt_args = Args(
        dataset_name=args.dataset_name,
        bottleneck_type=args.bottleneck_type,
        intervene=None,
        data_dir=None,
        attack_dir=None
    )  # only for extracting max_num_sentences and max_length, other variables ignored
    
    show_incorrect_only = False
    specified_annotation_id = None

#    if args.attack_dir is None:
#        rationale_pred_path = f'rr/predictions/{args.pred_dir}/original/{args.bottleneck_type}/rationale_predictions.json'
#        doc_dir = f'rr/base/explainable_qa/data/{args.dataset_name}/docs/'
#        rationale_dev_path = f'rr/base/explainable_qa/data/{args.dataset_name}/val.jsonl'
#    else:
#        rationale_pred_path = f'rr/predictions/{args.pred_dir}/{args.attack_dir}/{args.bottleneck_type}/rationale_predictions.json'
#        doc_dir = f'rr/attacks/data/{args.dataset_name}/{args.attack_dir}/docs/'
#        rationale_dev_path = f'rr/attacks/data/{args.dataset_name}/{args.attack_dir}/val.jsonl'
    print(f'dataset_name: {args.dataset_name}\nbottleneck_type: {args.bottleneck_type}\npred_dir: {args.pred_dir}\n')

    if args.dataset_name == 'fever':
        num_pos = 10
    elif args.dataset_name == 'multirc':
        num_pos = 14

    if args.run_original:
        rationale_pred_path = f'experiments/{args.pred_dir}/{args.bottleneck_type}/{args.model_name}/sentence_probabilities.txt'
#        rationale_pred_path = f'predictions/{args.pred_dir}/{args.bottleneck_type}/{args.model_name}/original/rationale_predictions.json'
        doc_dir = f'rr/base/explainable_qa/data/{args.dataset_name}/docs/'
        rationale_dev_path = f'rr/base/explainable_qa/data/{args.dataset_name}/val.jsonl'

        id_to_pred_gold_rationales = load_gold_and_pred_rationales(ckpt_args, rationale_pred_path, rationale_dev_path, doc_dir, debug=False, show_incorrect_only=show_incorrect_only, specified_annotation_id=specified_annotation_id)
    
        tps = []
        fps = []
        fns = []
        for annotation_id, blob in id_to_pred_gold_rationales.items():
            gold = set(blob['gold_sent_rationales'])
            pred = set(blob['pred_sent_rationales'])
    
            tp = len(gold & pred)
            fp = len(pred) - tp
            fn = len(gold) - tp
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
    
        # calc macro f1
        f1s = [2 * tp / (2 * tp + fp + fn) for tp, fp, fn in zip(tps, fps, fns)]
        macro_f1 = sum(f1s) / len(f1s)
        print(f'original | macro f1: {macro_f1 * 100:.2f}')
    
        # calc micro f1
#        micro_f1 = 2 * sum(tps) / (2 * sum(tps) + sum(fps) + sum(fns))
#        print('micro f1:', micro_f1)

    else:
        for attack_dir in [f'{args.attack_type}_pos{i}' for i in [0, 13]]:
            #rationale_pred_path = f'predictions/{args.pred_dir}/{args.bottleneck_type}/{args.model_name}/{attack_dir}/rationale_predictions.json'
            rationale_pred_path = f"predictions/{args.pred_dir}/{args.bottleneck_type}/{args.model_name}/{attack_dir}/rationale_predictions.json"
#            rationale_pred_path = f'predictions/{args.pred_dir}/{args.model_name}/{attack_dir}/{args.bottleneck_type}/rationale_predictions.json'
            doc_dir = f'rr/attacks/data/{args.dataset_name}/{attack_dir}/docs/'
            rationale_dev_path = f'rr/attacks/data/{args.dataset_name}/{attack_dir}/val.jsonl'

            id_to_pred_gold_rationales = load_gold_and_pred_rationales(ckpt_args, rationale_pred_path, rationale_dev_path, doc_dir, debug=False, show_incorrect_only=show_incorrect_only, specified_annotation_id=specified_annotation_id)
    
            tps = []
            fps = []
            fns = []
            for annotation_id, blob in id_to_pred_gold_rationales.items():
                gold = set(blob['gold_sent_rationales'])
                pred = set(blob['pred_sent_rationales'])
    
                tp = len(gold & pred)
                fp = len(pred) - tp
                fn = len(gold) - tp
                tps.append(tp)
                fps.append(fp)
                fns.append(fn)
    

            # calc macro f1
            f1s = [2 * tp / (2 * tp + fp + fn) for tp, fp, fn in zip(tps, fps, fns)]
            macro_f1 = sum(f1s) / len(f1s)
            print(f'{attack_dir} | macro f1: {macro_f1 * 100:.2f}')
    
            # calc micro f1
#            micro_f1 = 2 * sum(tps) / (2 * sum(tps) + sum(fps) + sum(fns))
#            print('micro f1:', micro_f1)
