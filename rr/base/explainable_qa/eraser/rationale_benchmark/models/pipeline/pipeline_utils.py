import itertools
import logging

from collections import defaultdict, namedtuple
from itertools import chain
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import classification_report, accuracy_score

from rationale_benchmark.metrics import (
    PositionScoredDocument,
    Rationale,
    partial_match_score,
    score_hard_rationale_predictions,
    score_soft_tokens
)

from rationale_benchmark.utils import Annotation

from rationale_benchmark.models.model_utils import PaddedSequence

SentenceEvidence = namedtuple('SentenceEvidence', 'kls ann_id query docid index sentence')

def annotations_to_evidence_classification(annotations: List[Annotation],
                                           documents: Dict[str, List[List[Any]]],
                                           class_interner: Dict[str, int],
                                           include_all: bool
                                          ) -> List[SentenceEvidence]:
    """Converts Corpus-Level annotations to Sentence Level relevance judgments.

    As this module is about a pipelined approach for evidence identification,
    inputs to both an evidence identifier and evidence classifier need to be to
    be on a sentence level, this module converts data to be that form.

    The return type is of the form
        annotation id -> docid -> [sentence level annotations]
    """
    ret = []
    for ann in annotations:
        ann_id = ann.annotation_id
        docids = set(ev.docid for ev in chain.from_iterable(ann.evidences))
        annotations_for_doc = defaultdict(list)
        for d in docids:
            for index, sent in enumerate(documents[d]):
                annotations_for_doc[d].append(
                    SentenceEvidence(
                        kls=class_interner[ann.classification],
                        query=ann.query,
                        ann_id=ann.annotation_id,
                        docid=d,
                        index=index,
                        sentence=tuple(sent)))
        if include_all:
            ret.extend(chain.from_iterable(annotations_for_doc.values()))
        else:
            contributes = set()
            for ev in chain.from_iterable(ann.evidences):
                for index in range(ev.start_sentence, ev.end_sentence):
                    contributes.add(annotations_for_doc[ev.docid][index])
            ret.extend(contributes)
    assert len(ret) > 0
    return ret

def annotations_to_evidence_identification(annotations: List[Annotation],
                                           documents: Dict[str, List[List[Any]]]
                                          ) -> Dict[str, Dict[str, List[SentenceEvidence]]]:
    """Converts Corpus-Level annotations to Sentence Level relevance judgments.

    As this module is about a pipelined approach for evidence identification,
    inputs to both an evidence identifier and evidence classifier need to be to
    be on a sentence level, this module converts data to be that form.

    The return type is of the form
        annotation id -> docid -> [sentence level annotations]
    """
    ret = defaultdict(dict) # annotation id -> docid -> sentences
    for ann in annotations:
        ann_id = ann.annotation_id
        for ev_group in ann.evidences:
            for ev in ev_group:
                if len(ev.text) == 0:
                    continue
                if ev.docid not in ret[ann_id]:
                    ret[ann.annotation_id][ev.docid] = []
                    # populate the document with "not evidence"; to be filled in later
                    for index, sent in enumerate(documents[ev.docid]):
                        ret[ann.annotation_id][ev.docid].append(SentenceEvidence(
                            kls=0,
                            query=ann.query,
                            ann_id=ann.annotation_id,
                            docid=ev.docid,
                            index=index,
                            sentence=sent))
                # define the evidence sections of the document
                for s in range(ev.start_sentence, ev.end_sentence):
                    ret[ann.annotation_id][ev.docid][s] = SentenceEvidence(
                            kls=1,
                            ann_id=ann.annotation_id,
                            query=ann.query,
                            docid=ev.docid,
                            index=ret[ann.annotation_id][ev.docid][s].index,
                            sentence=ret[ann.annotation_id][ev.docid][s].sentence)
    return ret

def make_preds_batch(classifier: nn.Module,
                     batch_elements: List[SentenceEvidence],
                     device: str=None,
                     criterion: nn.Module=None,
                     tensorize_model_inputs: bool=True) -> Tuple[float, List[float], List[int], List[int]]:
    """Batch predictions

    Args:
        classifier: a module that looks like an AttentiveClassifier
        batch_elements: a list of elements to make predictions over. These must be SentenceEvidence objects.
        device: Optional; what compute device this should run on
        criterion: Optional; a loss function
        tensorize_model_inputs: should we convert our data to tensors before passing it to the model? Useful if we have a model that performs its own tokenization
    """
    # delete any "None" padding, if any (imposed by the use of the "grouper")
    batch_elements = filter(lambda x: x is not None, batch_elements)
    targets, queries, sentences = zip(*[(s.kls, s.query, s.sentence) for s in batch_elements])
    ids = [(s.ann_id, s.docid, s.index) for s in batch_elements]
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    if tensorize_model_inputs:
        queries = [torch.tensor(q, dtype=torch.long) for q in queries]
        sentences = [torch.tensor(s, dtype=torch.long) for s in sentences]
        #queries = PaddedSequence.autopad(queries, device=device, batch_first=batch_first)
        #sentences = PaddedSequence.autopad(sentences, device=device, batch_first=batch_first)
    preds = classifier(queries, ids, sentences)
    targets = targets.to(device=preds.device)
    if criterion:
        loss = criterion(preds, targets)
    else:
        loss = None
    # .float() because pytorch 1.3 introduces a bug where argmax is unsupported for float16
    hard_preds = torch.argmax(preds.float(), dim=-1)
    return loss, preds, hard_preds, targets

def make_preds_epoch(classifier: nn.Module,
                      data: List[SentenceEvidence],
                      batch_size: int,
                      device: str=None,
                      criterion: nn.Module=None,
                      tensorize_model_inputs: bool=True):
    """Predictions for more than one batch.

    Args:
        classifier: a module that looks like an AttentiveClassifier
        data: a list of elements to make predictions over. These must be SentenceEvidence objects.
        batch_size: the biggest chunk we can fit in one batch.
        device: Optional; what compute device this should run on
        criterion: Optional; a loss function
        tensorize_model_inputs: should we convert our data to tensors before passing it to the model? Useful if we have a model that performs its own tokenization
    """
    epoch_loss = 0
    epoch_soft_pred = []
    epoch_hard_pred = []
    epoch_truth = []
    batches = _grouper(data, batch_size)
    classifier.eval()
    for batch in batches:
        loss, soft_preds, hard_preds, targets = make_preds_batch(classifier, batch, device, criterion=criterion, tensorize_model_inputs=tensorize_model_inputs)
        if loss is not None:
            epoch_loss += loss.sum().item()
        epoch_hard_pred.extend(hard_preds)
        epoch_soft_pred.extend(soft_preds.cpu())
        epoch_truth.extend(targets)
    epoch_loss /= len(data)
    epoch_hard_pred = [x.item() for x in epoch_hard_pred]
    epoch_truth = [x.item() for x in epoch_truth]
    return epoch_loss, epoch_soft_pred, epoch_hard_pred, epoch_truth

# copied from https://docs.python.org/3/library/itertools.html#itertools-recipes
def _grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def score_rationales(truth: List[Annotation],
                     documents: Dict[str, List[List[int]]],
                     input_data: List[SentenceEvidence],
                     scores: List[float]
                     ) -> dict:
    results = {}
    doc_to_sent_scores = dict() # (annid, docid) -> [sentence scores]
    for sent, score in zip(input_data, scores):
        k = (sent.ann_id, sent.docid)
        if k not in doc_to_sent_scores:
            doc_to_sent_scores[k] = [0.0 for _ in range(len(documents[sent.docid]))]
        doc_to_sent_scores[(sent.ann_id, sent.docid)][sent.index] = score[1].item()
    # hard rationale scoring
    best_sentence = {k:np.argmax(np.array(v)) for k,v in doc_to_sent_scores.items()}
    predicted_rationales = []
    for (ann_id, docid), sent_idx in best_sentence.items():
        start_token = sum(len(s) for s in documents[docid][:sent_idx])
        end_token = start_token + len(documents[docid][sent_idx])
        predicted_rationales.append(Rationale(ann_id, docid, start_token, end_token))
    true_rationales = list(chain.from_iterable(Rationale.from_annotation(rat) for rat in truth))

    results['hard_rationale_scores'] = score_hard_rationale_predictions(true_rationales, predicted_rationales)
    results['hard_rationale_partial_match_scores'] = partial_match_score(true_rationales, predicted_rationales, [0.5])

    # soft rationale scoring
    instance_format = []
    for (ann_id, docid), sentences in doc_to_sent_scores.items():
        soft_token_predictions = []
        for sent_score, sent_text in zip(sentences, documents[docid]):
            soft_token_predictions.extend(sent_score for _ in range(len(sent_text)))
        instance_format.append({
                                'annotation_id': ann_id,
                                'rationales': [{
                                    'docid': docid,
                                    'soft_rationale_predictions': soft_token_predictions,
                                    'soft_sentence_predictions': sentences,
                                    }],
                               })
    flattened_documents = {k:list(chain.from_iterable(v)) for k,v in documents.items()}
    token_scoring_format = PositionScoredDocument.from_results(instance_format, truth, flattened_documents, use_tokens=True)
    results['soft_token_scores'] = score_soft_tokens(token_scoring_format)
    sentence_scoring_format = PositionScoredDocument.from_results(instance_format, truth, documents, use_tokens=False)
    results['soft_sentence_scores'] = score_soft_tokens(sentence_scoring_format)
    return results

def decode(evidence_identifier: nn.Module,
           evidence_classifier: nn.Module,
           train: List[Annotation],
           val: List[Annotation],
           test: List[Annotation],
           docs: Dict[str, List[List[int]]],
           class_interner: Dict[str, int],
           batch_size: int,
           tensorize_model_inputs: bool,
           decoding_docs: Dict[str, List[Any]]=None) -> dict:
    """Identifies and then classifies evidence

    Args:
        evidence_identifier: a module for identifying evidence statements
        evidence_classifier: a module for making a classification based on evidence statements
        train: A List of interned Annotations
        val: A List of interned Annotations
        test: A List of interned Annotations
        docs: A Dict of Documents, which are interned sentences.
        class_interner: Converts an Annotation's final class into ints
        batch_size: how big should our batches be?
        tensorize_model_inputs: should we convert our data to tensors before passing it to the model? Useful if we have a model that performs its own tokenization
    """
    device = None
    class_labels = [k for k,v in sorted(class_interner.items(), key=lambda x: x[1])]
    if decoding_docs is None:
        decoding_docs = docs

    def prep(data: List[Annotation]) -> List[Tuple[SentenceEvidence, SentenceEvidence]]:
        """Prepares data for evidence identification and classification.

        Creates paired evaluation data, wherein each (annotation, docid, sentence, kls)
        tuplet appears first as the kls determining if the sentence is evidence, and
        secondarily what the overall classification for the (annotation/docid) pair is.
        This allows selection based on model scores of the evidence_identifier for
        input to the evidence_classifier.
        """
        identification_data = annotations_to_evidence_identification(data, docs)
        classification_data = annotations_to_evidence_classification(data, docs, class_interner, include_all=True)
        ann_doc_sents = defaultdict(lambda: defaultdict(dict)) # ann id -> docid -> sent idx -> sent data
        ret = []
        for sent_ev in classification_data:
            id_data = identification_data[sent_ev.ann_id][sent_ev.docid][sent_ev.index]
            ret.append((id_data, sent_ev))
            assert id_data.ann_id == sent_ev.ann_id
            assert id_data.docid == sent_ev.docid
            assert id_data.index == sent_ev.index
        assert len(ret) == len(classification_data)
        return ret

    def decode_batch(data: List[Tuple[SentenceEvidence, SentenceEvidence]], name: str, score: bool=False, annotations: List[Annotation]=None) -> dict:
        """Identifies evidence statements and then makes classifications based on it.

        Args:
            data: a paired list of SentenceEvidences, differing only in the kls field.
                  The first corresponds to whether or not something is evidence, and the second corresponds to an evidence class
            name: a name for a results dict
        """

        num_uniques = len(set((x.ann_id, x.docid) for x, _ in data))
        logging.info(f'Decoding dataset {name} with {len(data)} sentences, {num_uniques} annotations')
        identifier_data, classifier_data = zip(*data)
        results = dict()
        IdentificationClassificationResult = namedtuple('IdentificationClassificationResult', 'identification_data classification_data soft_identification hard_identification soft_classification hard_classification')
        with torch.no_grad():
            # make predictions for the evidence_identifier
            evidence_identifier.eval()
            evidence_classifier.eval()

            _, soft_identification_preds, hard_identification_preds, _ = make_preds_epoch(evidence_identifier, identifier_data, batch_size, device, tensorize_model_inputs=tensorize_model_inputs)
            assert len(soft_identification_preds) == len(data)
            identification_results = defaultdict(list)
            for id_data, cls_data, soft_id_pred, hard_id_pred in zip(identifier_data, classifier_data, soft_identification_preds, hard_identification_preds):
                res = IdentificationClassificationResult(identification_data=id_data,
                                                         classification_data=cls_data,
                                                         # 1 is p(evidence|sent,query)
                                                         soft_identification=soft_id_pred[1].float().item(),
                                                         hard_identification=hard_id_pred,
                                                         soft_classification=None,
                                                         hard_classification=False)
                identification_results[(id_data.ann_id, id_data.docid)].append(res)

            best_identification_results = {key:max(value, key=lambda x: x.soft_identification) for key, value in identification_results.items()}
            logging.info(f'Selected the best sentence for {len(identification_results)} examples from a total of {len(soft_identification_preds)} sentences')
            ids, classification_data = zip(*[(k, v.classification_data) for k,v in best_identification_results.items()])
            _, soft_classification_preds, hard_classification_preds, classification_truth = make_preds_epoch(evidence_classifier, classification_data, batch_size, device, tensorize_model_inputs=tensorize_model_inputs)
            classification_results = dict()
            for eyeD, soft_class, hard_class in zip(ids, soft_classification_preds, hard_classification_preds):
                input_id_result = best_identification_results[eyeD]
                res = IdentificationClassificationResult(identification_data=input_id_result.identification_data,
                                                         classification_data=input_id_result.classification_data,
                                                         soft_identification=input_id_result.soft_identification,
                                                         hard_identification=input_id_result.hard_identification,
                                                         soft_classification=soft_class,
                                                         hard_classification=hard_class)
                classification_results[eyeD] = res

            if score:
                truth = []
                pred = []
                for res in classification_results.values():
                    truth.append(res.classification_data.kls)
                    pred.append(res.hard_classification)
                #results[f'{name}_f1'] = classification_report(classification_truth, pred, target_names=class_labels, output_dict=True)
                results[f'{name}_f1'] = classification_report(classification_truth, hard_classification_preds, target_names=class_labels, output_dict=True)
                results[f'{name}_acc'] = accuracy_score(classification_truth, hard_classification_preds)
                results[f'{name}_rationale'] = score_rationales(annotations, decoding_docs, identifier_data, soft_identification_preds)


            # turn the above results into a format suitable for scoring via the rationale scorer
            # n.b. the sentence-level evidence predictions (hard and soft) are
            # broadcast to the token level for scoring. The comprehensiveness class
            # score is also a lie since the pipeline model above is faithful by
            # design.
            decoded = dict()
            decoded_scores = defaultdict(list)
            for (ann_id, docid), pred in classification_results.items():
                sentence_prediction_scores = [x.soft_identification for x in identification_results[(ann_id, docid)]]
                sentence_start_token = sum(len(s) for s in decoding_docs[docid][:pred.identification_data.index])
                sentence_end_token = sentence_start_token + len(decoding_docs[docid][pred.classification_data.index])
                hard_rationale_predictions = [{'start_token': sentence_start_token, 'end_token': sentence_end_token}]
                soft_rationale_predictions = []
                for sent_result in sorted(identification_results[(ann_id, docid)], key=lambda x: x.identification_data.index):
                    soft_rationale_predictions.extend(sent_result.soft_identification for _ in range(len(decoding_docs[sent_result.identification_data.docid][sent_result.identification_data.index])))
                if ann_id not in decoded:
                    decoded[ann_id] = {
                        "annotation_id": ann_id,
                        "rationales": [],
                        "classification": class_labels[pred.hard_classification],
                        "classification_scores": {class_labels[i]:s.item() for i, s in enumerate(pred.soft_classification)},
                        # TODO this should turn into the data distribution for the predicted class
                        #"comprehensiveness_classification_scores": 0.0,
                        "truth": pred.classification_data.kls,
                    }
                decoded[ann_id]['rationales'].append({
                            "docid": docid,
                            "hard_rationale_predictions": hard_rationale_predictions,
                            "soft_rationale_predictions": soft_rationale_predictions,
                            "soft_sentence_predictions": sentence_prediction_scores,
                        })
                decoded_scores[ann_id].append(pred.soft_classification)

            # in practice, this is always a single element operation:
            # in evidence inference (prompt is really a prompt + document), fever (we split documents into two classifications), movies (you only have one opinion about a movie), or boolQ (single document prompts)
            # this exists to support weird models we *might* implement for cose/esnli
            for ann_id, scores_list in decoded_scores.items():
                scores = torch.stack(scores_list)
                score_avg = torch.mean(scores, dim=0)
                # .float() because pytorch 1.3 introduces a bug where argmax is unsupported for float16
                hard_pred = torch.argmax(score_avg.float()).item()
                decoded[ann_id]['classification'] = class_labels[hard_pred]
                decoded[ann_id]['classification_scores'] = {class_labels[i]:s.item() for i,s in enumerate(score_avg)}
            return results, list(decoded.values())

    test_results, test_decoded = decode_batch(prep(test), 'test', score=False)
    val_results, val_decoded = dict(), []
    train_results, train_decoded = dict(), []
    #val_results, val_decoded = decode_batch(prep(val), 'val', score=True, annotations=val)
    #train_results, train_decoded = decode_batch(prep(train), 'train', score=True, annotations=train)
    return dict(**train_results, **val_results, **test_results), train_decoded, val_decoded, test_decoded

