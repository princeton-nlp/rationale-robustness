import torch

from rrtl.config import Config

config = Config()


def log(mode, args, log_dict):
    if mode == 'train':
        print((
            f"[train] Epoch: {log_dict['epoch']} | "
            f"batch: {log_dict['batch_idx']} / {log_dict['num_batches']} (global step: {log_dict['global_step']}) | "
            f"train acc: {log_dict['acc'] * 100:.2f} | "
            f"train loss: {log_dict['loss']:.4f} | "
            f"Elapsed {log_dict['elapsed']:.2f}s"
        ))
    elif mode in ('dev', 'test'):
        print((
            f"[{mode}] Epoch: {log_dict['epoch']} | "
            f"global step: {log_dict['global_step']} | "
            f"{mode} acc: {log_dict['eval_acc'] * 100:.2f} | "
            f"{mode} loss: {log_dict['eval_loss']:.4f}"
        ))


def log_qa(mode, args, log_dict):
    if mode == 'train':
        print((
            f"[train] Epoch: {log_dict['epoch']} | "
            f"batch: {log_dict['batch_idx']} / {log_dict['num_batches']} (global step: {log_dict['global_step']}) | "
            f"loss: {log_dict['loss']:.4f} | "
            f"pred loss: {log_dict['train_pred_loss']:.4f} | "
            f"KL loss: {log_dict['train_kl_loss']:.4f} | "
            f"Rationale loss: {log_dict['train_sent_rationale_loss']:.4f} | "
            f"Neg Rationale loss: {log_dict['train_neg_rationale_loss']:.4f} | "
            f"EM: {log_dict['train_em'] * 100:.2f} | "
            f"F1: {log_dict['train_f1'] * 100:.2f} | "
            f"Elapsed {log_dict['elapsed']:.2f}s"
        ))
    elif mode in ('dev', 'test'):
        print((
            f"[{mode}] Epoch: {log_dict['epoch']} | "
            f"global step: {log_dict['global_step']} | "
            f"EM: {log_dict['eval_em'] * 100:.2f} | "
            f"F1: {log_dict['eval_f1'] * 100:.2f} | "
            f"loss: {log_dict['eval_loss']:.4f}"
        ))


def visualize_rationale(args, batch, output, tokenizer):
    if args.dataset_name == 'esnli':
        ids = batch['input_ids'].tolist()
        id_to_labelname = {v: k for k, v in config.NLI_LABEL.items()}
        pred_labels = torch.argmax(output['logits'], dim=1).tolist()
        labels = batch['labels'].tolist()
        gold_rationales = batch['rationales'].long().tolist()

        for ids, rationale, gold_rationale, label, pred_label in zip(batch['input_ids'].tolist(), output['rationales'], gold_rationales, labels, pred_labels):
            tokens = tokenizer.convert_ids_to_tokens(ids)
            rationale = rationale.long().tolist()
            highlight = [(token, r, gr) for token, r, gr in zip(tokens, rationale, gold_rationale) if token != '[PAD]']
            pred_highlight = [f'*{token}*' if r == 1 and token not in ('[CLS]', '[SEP]') else f'{token}' for token, r, gr in highlight]
            gold_highlight = [f'*{token}*' if gr == 1 else f'{token}' for token, r, gr in highlight]
            print('[Gold]:', id_to_labelname[label])
            print('[Pred]:', id_to_labelname[pred_label])
            print('[Gold rationale]:', ' '.join(gold_highlight))
            print('[Pred rationale]:', ' '.join(pred_highlight))
            print('---')
            input()
    elif args.dataset_name in ('cad', 'hans', 'esnli_attack'):
        ids = batch['input_ids'].tolist()
        id_to_labelname = {v: k for k, v in config.NLI_LABEL.items()}
        pred_labels = torch.argmax(output['logits'], dim=1).tolist()
        labels = batch['labels'].tolist()

        for ids, rationale, label, pred_label in zip(batch['input_ids'].tolist(), output['rationales'], labels, pred_labels):
            tokens = tokenizer.convert_ids_to_tokens(ids)
            rationale = rationale.long().tolist()
            highlight = [(token, r) for token, r in zip(tokens, rationale) if token != '[PAD]']
            pred_highlight = [f'*{token}*' if r == 1 and token not in ('[CLS]', '[SEP]') else f'{token}' for token, r in highlight]
            print('[Gold]:', id_to_labelname[label])
            print('[Pred]:', id_to_labelname[pred_label])
            print('[Pred rationale]:', ' '.join(pred_highlight))
            print('---')
            input()