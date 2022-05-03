import torch

from rrtl.config import Config

config = Config()


def gold_rationale_capture_rate(args, batch, output, tokenizer):
    tps = []
    fps = []
    fns = []

    if args.dataset_name == 'esnli':
        ids = batch['input_ids'].tolist()
        id_to_labelname = {v: k for k, v in config.NLI_LABEL.items()}
        pred_labels = torch.argmax(output['logits'], dim=1).tolist()
        labels = batch['labels'].tolist()
        gold_rationales = batch['rationales'].long().tolist()

        for ids, rationale, gold_rationale, label, pred_label in zip(batch['input_ids'].tolist(), output['rationales'], gold_rationales, labels, pred_labels):
            tokens = tokenizer.convert_ids_to_tokens(ids)
            rationale = rationale.long().tolist()
            pred = set()
            gold = set()
            for token_pos, (token, r, gr) in enumerate(zip(tokens, rationale, gold_rationale)):
                if token in ('[CLS]', '[SEP]', '[PAD]'):
                    continue
                if r == 1:
                    pred.add(token_pos)
                if gr == 1:
                    gold.add(token_pos)
            tp = len(gold & pred)
            fp = len(pred) - tp
            fn = len(gold) - tp
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
    return tps, fps, fns