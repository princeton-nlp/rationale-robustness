import sys; sys.path.insert(0, "..")
import argparse
import json
import logging
import random
import os
import glob
import pdb
from pathlib import Path

from itertools import chain
from typing import Set

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report

from eraser.rationale_benchmark.utils import (
    write_jsonl,
    load_datasets,
    load_documents,
    annotations_from_jsonl,
    intern_documents,
    intern_annotations
)

from ib_utils import read_examples, convert_examples_to_sentence_features
#from evidence_utils import (
#    read_examples,
#    convert_examples_to_features,
#    convert_binary_mask_rationales,
#    score_hard_rationale_predictions
#)

from transformers import (WEIGHTS_NAME, BertTokenizer, RobertaTokenizer, BertForSequenceClassification, \
                          BertConfig, RobertaConfig, \
                          RobertaForSequenceClassification, BertForMultipleChoice, RobertaForMultipleChoice,
                          AdamW, get_linear_schedule_with_warmup)

from baseline_models import BertForTaskCumEvidenceClassfication

import wandb

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
# let's make this more or less deterministic (not resistent to restarts)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta' : (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'multitask_bert' : (BertConfig, BertForTaskCumEvidenceClassfication, BertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def load_and_cache_examples(args, model_params, tokenizer, evaluate=False, split="train", output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # only load one split
    input_file = os.path.join(args.data_dir, split)
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}_multitask'.format(
        split,
        list(filter(None, model_params["tokenizer_name"].split('/'))).pop(),
        str(args.max_seq_length)))
    if args.gold_evidence:
        cached_features_file += "_goldevidence"

    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        dataset = annotations_from_jsonl(os.path.join(args.data_dir, split + ".jsonl"))

        docids = set(e.docid for e in
                     chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(dataset)))))
        documents = load_documents(args.data_dir, docids)
        
        if args.out_domain:
            examples = read_json(args)
        else:
            examples = read_examples(args, model_params, dataset, documents, split)

        features = convert_examples_to_sentence_features(
            args,
            model_params,
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            max_query_length=args.max_query_length,
            is_training=not evaluate
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    # Tensorize all features
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_unique_ids = torch.tensor([f.unique_id for f in features], dtype=torch.float)

    all_sentence_starts = torch.tensor([f.sentence_starts for f in features], dtype=torch.long)
    all_sentence_ends = torch.tensor([f.sentence_starts for f in features], dtype=torch.long)
    all_sentence_mask = torch.tensor([f.sentence_mask for f in features], dtype=torch.long)
    all_evidence_labels = torch.tensor([f.evidence_label for f in features], dtype=torch.long)
    all_neg_rationales = torch.tensor([f.neg_rationales if hasattr(f, 'neg_rationales') else [0] * len(f.evidence_label) for f in features], dtype=torch.long)

    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        tensorized_dataset = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_example_index,
            all_cls_index,
            all_p_mask,
            all_unique_ids,
            all_sentence_starts,
            all_sentence_ends,
            all_sentence_mask,
            all_evidence_labels,
        )
    else:
        all_class_labels = torch.tensor([f.class_label for f in features], dtype=torch.long)
#        all_evidence_labels = torch.tensor([f.evidence_label for f in features], dtype=torch.long)
        tensorized_dataset = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_class_labels,
            all_cls_index,
            all_p_mask,
            all_unique_ids,
            all_sentence_starts,
            all_sentence_ends,
            all_sentence_mask,
            all_evidence_labels,
            all_neg_rationales
        )

    if output_examples:
        return tensorized_dataset, examples, features
    return tensorized_dataset, features

def train(args, model_params, train_dataset, model, tokenizer):
#    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#    sent_rationale_file = os.path.join(args.output_dir, "sent_rationale_labels.txt")
#    sent_rationale_file_pointer = open(sent_rationale_file, "w")

#    if args.local_rank in [-1, 0] and args.tf_summary:
#        tb_writer = SummaryWriter("runs/" + os.path.basename(args.output_dir))
    if args.local_rank in [-1, 0] and args.wandb:
        wandb.init(project='rationale-robustness-origin', name=args.model_name)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    if args.evaluate_during_training:
        dataset, eval_features = load_and_cache_examples(args, model_params, tokenizer, evaluate=True, split="val", output_examples=False)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_f1 = (-1, -1) # Classfication F1, accuracy ; Accuracy of evidence prediction
    wait_step = 0
    stop_training = False
    metric_name = "F1"
    epoch = 0
    for _ in train_iterator:
#    for i_epoch, _ in enumerate(train_iterator):
#        if i_epoch != 0:
#            sent_rationale_file_pointer.close()
#            dsdsds
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1],
                      'token_type_ids':  None if args.model_type == 'xlm' or args.model_type == 'roberta' else batch[2],
                      'labels': batch[3]}
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask': batch[5]})
            inputs.update({
                "sentence_starts" : batch[7],
                "sentence_ends" : batch[8],
                "sentence_mask": batch[9],
                'evidence_labels': batch[10],
                'neg_rationales': batch[11]
            })

#            batch_sent_rationales = batch[-1].tolist()
#            batch_sentence_mask = batch[-2].tolist()
#            for sent_rationales, sent_mask in zip(batch_sent_rationales, batch_sentence_mask):
#                sent_rationale_file_pointer.write(str(sent_rationales) + '<SEP>' + str(sent_mask) + '\n')
#            continue
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            evidence_loss = outputs[1]
            neg_rationale_loss = outputs[2]
            # combined both losses
            loss += args.gamma * evidence_loss
            loss += neg_rationale_loss
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = predict(args, model_params, model, tokenizer, eval_features, eval_dataloader, global_step)
                        #for key, value in results.items():
                        #    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        if best_f1 < results:
                            logger.info("Saving model with best %s: %.2f (Acc %.2f) -> %.2f (Acc %.2f) on epoch=%d" % \
                                        (metric_name, best_f1[0] * 100, best_f1[1] * 100, results[0] * 100, results[1] * 100,
                                         epoch))
                            output_dir = os.path.join(args.output_dir, 'best_model')
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            best_f1 = results
                            wait_step = 0
                            stop_training = False
                        else:
                            wait_step += 1
                            if wait_step == args.wait_step:
                                logger.info("Loosing Patience")
                                stop_training = True
                    #if args.tf_summary:
                    #    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    #    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    if args.wandb:
                        log_dict = {
                            'lr': scheduler.get_lr()[0],
                            'loss': (tr_loss - logging_loss) / args.logging_steps,
                            'global_step': global_step,
                        }
                        wandb.log(log_dict)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if stop_training or args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if stop_training or args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        epoch += 1
    #if args.local_rank in [-1, 0] and args.tf_summary:
    #    tb_writer.close()

    return global_step, tr_loss / global_step

def predict(args, model_params, model, tokenizer, eval_features, eval_dataloader, global_step):
    all_results = []
    all_targets = []
    all_rationale_results = []
    all_rationale_targets = []
    results = None
    class_interner = dict((y, x) for (x, y) in enumerate(model_params['classes']))
    class_labels = [k for k, v in sorted(class_interner.items(), key=lambda x: x[1])]
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        mask_sums = to_list(torch.sum(batch[1], -1) - torch.sum(batch[1][:, :args.max_query_length], -1))
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type == 'xlm' or args.model_type == 'roberta' else batch[2]
                      # XLM don't use segment_ids
                      }
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask': batch[5]})
            inputs.update({
                "sentence_starts" : batch[-4],
                "sentence_ends" : batch[-3],
                "sentence_mask": batch[-2],
                'evidence_labels': batch[-1]
            })
            outputs = model(**inputs)
            class_logits = outputs[1]  # index 0 is evidence_loss
            evidence_logits = outputs[2]
            hard_preds = to_list(torch.argmax(class_logits.float(), dim=-1))
#            hard_evidence_preds = to_list(torch.argmax(evidence_logits.float(), dim=-1))
            all_results.extend(hard_preds)
            # create a list of Rationales for contiguous 1s (with annotation and document id)
#            all_rationale_results.extend([item for m_len, sublist in zip(mask_sums, hard_evidence_preds) for item in sublist[:m_len]])
        for i, example_index in enumerate(example_indices):
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_targets.extend([eval_feature.class_label])
            evidence_label = list(eval_feature.evidence_label[args.max_query_length:][:mask_sums[i]])
            all_rationale_targets.extend(evidence_label)
    results = (classification_report(all_targets, all_results, target_names=class_labels, output_dict=True)['weighted avg']['f1-score'], accuracy_score(all_targets, all_results))
#    evidence_report = classification_report(all_rationale_targets, all_rationale_results, target_names=["POS", "NEG"], output_dict=True)
#    evidence_results = evidence_report['weighted avg']['f1-score'] #(evidence_report['NEG']['f1-score'], evidence_report['POS']['f1-score'])
    #results = results + (evidence_results,)
    # combined_results = ((results[0] + evidence_results)/2, results[1])
    combined_results = (results[0], results[1])
    if args.wandb:
        log_dict = {
            'task_perf': results[0],
            'global_step': global_step,
        }
        wandb.log(log_dict)
    return combined_results


def evaluate(args, model_params, model, tokenizer, prefix="", output_examples=False, split="val"):
    full_pred_dir = os.path.join(
        args.base_pred_dir,
        args.pred_dir,
        args.bottleneck_type,
        args.model_name,
        'original'
    )  # NOTE: output_dir replaced by pred_dir here
    if output_examples:
        dataset, examples, features = load_and_cache_examples(args, model_params, tokenizer, evaluate=True, split=split, output_examples=output_examples)
    else:
        dataset, features = load_and_cache_examples(args, model_params, tokenizer, evaluate=True, split=split, output_examples=output_examples)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    all_targets = []
    results = []
    class_interner = dict((y, x) for (x, y) in enumerate(model_params['classes']))
    class_labels = [k for k, v in sorted(class_interner.items(), key=lambda x: x[1])]
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        mask_sums = to_list(torch.sum(batch[1], -1) - torch.sum(batch[1][:, :args.max_query_length], -1))
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type == 'xlm' or args.model_type == 'roberta' else batch[2]  # XLM don't use segment_ids
                      }
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask':    batch[5]})
            inputs.update({
                "sentence_starts" : batch[-4],
                "sentence_ends" : batch[-3],
                "sentence_mask": batch[-2],
                'evidence_labels': batch[-1]
            })
            outputs = model(**inputs)
            logits = outputs[1]  # index 0 is evidence_loss
            hard_preds = to_list(torch.argmax(logits.float(), dim=-1))
            all_results.extend(hard_preds)
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_targets.extend([eval_feature.class_label])
    results = (classification_report(all_targets, all_results, target_names=class_labels, output_dict=True)['weighted avg']['f1-score'], accuracy_score(all_targets, all_results))
    logger.info('Classification Report: {}'.format(classification_report(all_targets, all_results, target_names=class_labels, output_dict=True)))
    return results

def put_args_to_model_params(args):
    return vars(args)

def main():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', required=True,
                        help='Which directory contains a {train,val,test}.jsonl file?')
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='Where shall we write intermediate models + final data to?')
#    parser.add_argument('--model_params', dest='model_params', required=True,
#                        help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--debug", action="store_true", default=False)
    #parser.add_argument("--tf_summary", action="store_true", default=False)
    parser.add_argument("--out_domain", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--model_name", type=str, default="try")

    # Input parameters
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int)
    # Variants of baselines that changes what input is loaded
    parser.add_argument("--full_doc", action="store_true", default=False)
    parser.add_argument("--gold_evidence", action="store_true", default=False)
    parser.add_argument("--multitask", action="store_true", default=False)
    parser.add_argument("--predicted_train_evidence_file", type=str, default=None)
    parser.add_argument("--predicted_eval_evidence_file", type=str, default=None)
    parser.add_argument("--random_evidence", action="store_true", default=False)
    parser.add_argument("--gamma", type=float, default=1.0, help="How much gold to feed to the supervision branch")


    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--wait_step", default=5, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=50.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    # Logging
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # Multi-GPU
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")


    # param for ib utils
    parser.add_argument("--truncate", default=False, action="store_true")
    parser.add_argument('--low_resource', action="store_true", default=False)
    parser.add_argument("--max_num_sentences", default=10, type=int)

    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument('--config_name', type=str, default='bert-base-uncased')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased')
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--classes', type=str, nargs='+', required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument("--bottleneck-type", type=str, required=True, help="[vib | vib_semi | full | full_multitask]")
    parser.add_argument("--model-name", type=str, default="try")
    parser.add_argument("--pred-dir", type=str, required=True, help="E.g., fever_beam_search")
    parser.add_argument("--base-pred-dir", type=str, default="/n/fs/nlp-hc22/rationale-robustness/predictions")
    parser.add_argument("--use_neg_rationales", action="store_true")


    args = parser.parse_args()

    # Parse model args json
#    with open(args.model_params, 'r') as fp:
#        logging.debug(f'Loading model parameters from {args.model_params}')
#        model_params = json.load(fp)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    if args.do_train and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    model_params = put_args_to_model_params(args)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        f.write(json.dumps(model_params, sort_keys=True, indent=4))
    
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(model_params["config_name"] if model_params["config_name"] else model_params["model_name_or_path"])
    config.num_labels = len(model_params['classes'])
    config.max_query_length = args.max_query_length
    config.use_neg_rationales = args.use_neg_rationales
    tokenizer = tokenizer_class.from_pretrained(model_params["tokenizer_name"] if model_params["tokenizer_name"] else model_params["model_name_or_path"],
                                                do_lower_case=model_params["do_lower_case"])
    model = model_class.from_pretrained(model_params["model_name_or_path"], from_tf=bool('.ckpt' in model_params["model_name_or_path"]),
                                        config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        train_dataset, _ = load_and_cache_examples(args, model_params, tokenizer, evaluate=False, split="train", output_examples=False)
        global_step, tr_loss = train(args, model_params, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=model_params["do_lower_case"])
        model.to(args.device)

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir + "/best_model"]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model_params, model, tokenizer, prefix=global_step, output_examples=True, split=args.eval_split)
            results = {"Best F1":result[0], "Best Accuracy":result[1]}

    logger.info("Results on the split {} : {}".format(args.eval_split, results))
    return results

if __name__ == '__main__':
    main()
    #
    # # Loading annotations

    # # create a vocabulary from whitespace tokenization
    # document_vocab = set(chain.from_iterable(chain.from_iterable(documents.values())))
    # annotation_vocab = set(chain.from_iterable(e.query.split() for e in chain(train, val, test)))
    # logging.debug(f'Loaded {len(documents)} documents with {len(document_vocab)} unique words')
    #
    # # Union : this ignores the case where annotations don't align perfectly with token boundaries, but this isn't that important
    # vocab = document_vocab | annotation_vocab
    # unk_token = 'UNK'
    #
    # # Loading only rationales vs loading full documents into interned (tensorized) variables
    # classifier, tokenizer, predictions = initialize_models(model_params, vocab, batch_first=BATCH_FIRST, unk_token=unk_token)





