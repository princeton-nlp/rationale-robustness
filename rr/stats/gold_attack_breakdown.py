import argparse

from rr.eval.model_args import Args
from rr.eval.utils import load_gold_and_pred_rationales
from rr.config import Config

config = Config()


def matrix_sum(m):
    return sum([sum(row) for row in m])


if __name__ == '__main__':
    """
    fever vib:
        python -m rr.stats.gold_attack_breakdown --dataset-name fever --pred-dir fever_beam_search --model-name fever_vib_pi=0.4_beta=1.0 --bottleneck-type vib
    fever vib-sup:
        python -m rr.stats.gold_attack_breakdown --dataset-name fever --pred-dir fever_beam_search --model-name fever_vib_semi_pi=0.4_beta=0.0_gamma=2.0 --bottleneck-type vib_semi
    fever vib-sup neg_r:
        python -m rr.stats.gold_attack_breakdown --dataset-name fever --pred-dir try --model-name fever_vib_semi_pi=0.4_beta=0.1_gamma=2.0_neg_r --bottleneck-type vib_semi
    fever vib neg_r:
        python -m rr.stats.gold_attack_breakdown --dataset-name fever --pred-dir try --model-name fever_vib_pi=0.4_beta=2.0_neg_r --bottleneck-type vib
    multirc vib:
        python -m rr.stats.gold_attack_breakdown --dataset-name multirc --pred-dir multirc_beam_search --model-name multirc_vib_pi=0.4_beta=1.0 --bottleneck-type vib
    multirc vib-sup:
        python -m rr.stats.gold_attack_breakdown --dataset-name multirc --pred-dir multirc_beam_search --model-name multirc_vib_semi_pi=0.2_beta=0.0_gamma=1.0 --bottleneck-type vib_semi

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True, help="[fever | multirc]")
    parser.add_argument("--bottleneck-type", type=str, required=True, help="[vib | vib_semi | full | full_multitask]")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--pred-dir", type=str, default=None, help="[None | fever_pi0.4 | ...]. If `None` then use `dataset_name`.")
    parser.add_argument("--max_pos", type=int, default=6, help="Max attack position to print out.")
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
    print(f'dataset_name: {args.dataset_name}\nbottleneck_type: {args.bottleneck_type}\npred_dir: {args.pred_dir}\n')
    target_vocab = config.target_vocab[args.dataset_name]

    for i in range(6):
        attack_dir = f'addsent_pos{i}'
        rationale_pred_path = f'predictions/{args.pred_dir}/{args.bottleneck_type}/{args.model_name}/{attack_dir}/rationale_predictions.json'
        doc_dir = f'rr/attacks/data/{args.dataset_name}/{attack_dir}/docs/'
        rationale_dev_path = f'rr/attacks/data/{args.dataset_name}/{attack_dir}/val.jsonl'

        id_to_pred_gold_rationales = load_gold_and_pred_rationales(ckpt_args,
                                                                   rationale_pred_path,
                                                                   rationale_dev_path,
                                                                   doc_dir,
                                                                   debug=False,
                                                                   show_incorrect_only=show_incorrect_only,
                                                                   specified_annotation_id=specified_annotation_id)

        corrects = []
        # confusion matrix
        #        target A, target B
        # pred A      ...       ...
        # pred B      ...       ...
        # FEVER: A = REFUTES, B = SUPPORT
        # MultiRC: A = True, B = False
        g_a = [[0, 0], [0, 0]]
        g_na = [[0, 0], [0, 0]]
        ng_a = [[0, 0], [0, 0]]
        ng_na = [[0, 0], [0, 0]]

        for annotation_id, blob in id_to_pred_gold_rationales.items():
            pred_target = blob['pred_target']
            gold_target = blob['gold_target']
            attack = {i}
            gold = set(blob['gold_sent_rationales'])
            pred = set(blob['pred_sent_rationales'])

            corrects.append(pred_target == gold_target)
            if len(gold & pred) and len(attack & pred):
                g_a[target_vocab[pred_target]][target_vocab[gold_target]] += 1
            elif len(gold & pred) and not len(attack & pred):
                g_na[target_vocab[pred_target]][target_vocab[gold_target]] += 1
            elif not len(gold & pred) and len(attack & pred):
                ng_a[target_vocab[pred_target]][target_vocab[gold_target]] += 1
            elif not len(gold & pred) and not len(attack & pred):
                ng_na[target_vocab[pred_target]][target_vocab[gold_target]] += 1

        print(f'{attack_dir}:')
        acc = sum(corrects) / len(corrects)
        print(f'accuracy            = {acc * 100:.2f} | total = {len(corrects):>4}')

        total_g_a = matrix_sum(g_a)
        if total_g_a:
            g_a_acc = (g_a[0][0] + g_a[1][1]) / total_g_a
            print(f'gold + attack       = {g_a_acc * 100:.2f} | total = {total_g_a:>4} ({total_g_a / len(corrects) * 100:.1f}%)')

        total_g_na = matrix_sum(g_na)
        if total_g_na:
            g_na_acc = (g_na[0][0] + g_na[1][1]) / total_g_na
            print(f'gold + no attack    = {g_na_acc * 100:.2f} | total = {total_g_na:>4} ({total_g_na / len(corrects) * 100:.1f}%)')
            #print( '                    ---------------')
            #print(f'                    | {g_na[0][0]:>4} | {g_na[0][1]:>4} |')
            #print(f'                    | {g_na[1][0]:>4} | {g_na[1][1]:>4} |')
            #print( '                    ---------------')

        total_ng_a = matrix_sum(ng_a)
        if total_ng_a:
            ng_a_acc = (ng_a[0][0] + ng_a[1][1]) / total_ng_a
            print(f'no gold + attack    = {ng_a_acc * 100:.2f} | total = {total_ng_a:>4} ({total_ng_a / len(corrects) * 100:.1f}%)')
            #print( '                    ---------------')
            #print(f'                    | {ng_a[0][0]:>4} | {ng_a[0][1]:>4} |')
            #print(f'                    | {ng_a[1][0]:>4} | {ng_a[1][1]:>4} |')
            #print( '                    ---------------')

        total_ng_na = matrix_sum(ng_na)
        if total_ng_na:
            ng_na_acc = (ng_na[0][0] + ng_na[1][1]) / total_ng_na
            print(f'no gold + no attack = {ng_na_acc * 100:.2f} | total = {total_ng_na:>4} ({total_ng_na / len(corrects) * 100:.1f}%)')
            #print( '                    ---------------')
            #print(f'                    | {ng_na[0][0]:>4} | {ng_na[0][1]:>4} |')
            #print(f'                    | {ng_na[1][0]:>4} | {ng_na[1][1]:>4} |')
            #print( '                    ---------------')
        print()
        print( '     G & A            G & NA          NG & A           NG & NA    ')
        print( '---------------  ---------------  ---------------  ---------------')
        print(f'| {g_a[0][0]:>4} | {g_a[0][1]:>4} |  | {g_na[0][0]:>4} | {g_na[0][1]:>4} |  | {ng_a[0][0]:>4} | {ng_a[0][1]:>4} |  | {ng_na[0][0]:>4} | {ng_na[0][1]:>4} |')
        print(f'| {g_a[1][0]:>4} | {g_a[1][1]:>4} |  | {g_na[1][0]:>4} | {g_na[1][1]:>4} |  | {ng_a[1][0]:>4} | {ng_a[1][1]:>4} |  | {ng_na[1][0]:>4} | {ng_na[1][1]:>4} |')
        print( '---------------  ---------------  ---------------  ---------------')
        print('\n')
    
    original_id_to_pred_gold_rationales = load_gold_and_pred_rationales(ckpt_args,
                                                                        f'predictions/{args.pred_dir}/{args.bottleneck_type}/{args.model_name}/original/rationale_predictions.json',
                                                                        f'rr/base/explainable_qa/data/{args.dataset_name}/val.jsonl',
                                                                        f'rr/base/explainable_qa/data/{args.dataset_name}/docs/',
                                                                        debug=False,
                                                                        show_incorrect_only=show_incorrect_only,
                                                                        specified_annotation_id=specified_annotation_id)
    orig_corrects = []
    for annotation_id, blob in original_id_to_pred_gold_rationales.items():
        pred_target = blob['pred_target']
        gold_target = blob['gold_target']
        orig_corrects.append(pred_target == gold_target)
    orig_acc = sum(orig_corrects) / len(orig_corrects)
    print(f'original accuracy = {orig_acc * 100:.2f} | total = {len(orig_corrects)}')
    
