import os
import glob
import json
import random
from collections import defaultdict
from pathlib import Path
import argparse
from argparse import Namespace
import re

from nltk.corpus import words
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import numpy as np
import spacy
import pickle
import editdistance

from rr.config import Config 

import rr.attacks.convert_queries as convert_queries
import rr.attacks.corenlp as corenlp
CORENLP_LOG = 'corenlp.log'
CORENLP_PORT = 8000

SAVE_PATH = 'your/path/to/saved/attacks'

nlp = spacy.load("en_core_web_lg")
config = Config()

# Pull nearby GLoVE vectors
NEARBY_GLOVE_FILE = 'your/glove/base/nearest/neighbor/save/path'
nearby_word_dict = {}

dataset_ents = defaultdict(list)  #dict 


def extract_ents(query):
    doc = nlp(query)
    for ent in doc.ents:
        if ent.text.replace("||", "") not in dataset_ents[ent.label_]:
            dataset_ents.setdefault(ent.label_,[]).append(ent.text.replace("||", ""))


def load_random_sentences_from_wikitext103(k):
    sentences = []
    with open(config.WIKITEXT103_DEV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentences += line.split('.')
    random.shuffle(sentences)
    return sentences[:k]


def load_random_sentences(k):
    """
    k: number of sentences to generate
    """
    sentences = []
    for i in range(k):
        sent = ' '.join(random.sample(words.words(), 10))
        sentences.append(sent)
    return sentences


def replace_nearby_words(sent):
    changed = []
    parsed = nlp(sent.lower())
    for token in parsed:
        if token.pos_ in ('NOUN', 'VERB', 'NUM', 'PROPN', 'INTJ'):
            ## change
            if token.text in nearby_word_dict:
                i = 0
                nearest_list = nearby_word_dict[token.text]
                nearest = nearest_list[i]["word"]
                while editdistance.eval(nearest, token.text) < 3 and i < len(nearest_list)-1:
                    i += 1
                    nearest = nearby_word_dict[token.text][i]["word"]
                changed.append(nearest)
            elif token.pos_ in ('NOUN', 'PROPN'):
                rand_word = random.choice(list(nearby_word_dict))
                while nlp(rand_word)[0].pos_ not in ('NOUN', 'PROPN'):
                    rand_word = random.choice(list(nearby_word_dict))
                changed.append(rand_word)
            else:
                changed.append(token.text) # should never reach here
        else:
            changed.append(token.text)
    return ' '.join(changed).strip()


def addsent_multirc(query):
    print("original: ", query)
    split = query.split("||", maxsplit=1)
    question = split[0]
    answer = split[1]

    # 1. Addsent attack on question / answer, separately
    changed_question = addsent_attack(question, obj.classification)
    changed_answer = addsent_attack(answer, obj.classification)

    # 1. Did the answer change? 
    if (changed_answer.strip() == answer.lower().strip()):
        changed_answer = replace_nearby_words(changed_answer)

    # 2. Did the question change? 
    if (changed_question.strip() == question.lower().strip()): 
        changed_question = replace_nearby_words(changed_question)

    ch_query = changed_question + " || " + changed_answer
    return ch_query


def addsent_attack(query, classification):
    o_query = query
    attacked_query = []
    orig_query = []

    # 0. replace named entities
    flipped = False
    if classification == "SUPPORTS" and args.aug_method == 'addsent-strong':
        if " is " in query:
            query = query.replace(" is ", " is not ")
            flipped = True
        elif " is not " in query:
            query = query.replace(" is not ", " is ")
            flipped = True
        
        if " was " in query:
            query = query.replace(" was ", " was not ")
            flipped = True
        elif " was not " in query:
            query = query.replace(" was not ", " was ")
            flipped = True
    
    query_ents = nlp(query).ents
    for ent in query_ents:
        replacement = random.choice(dataset_ents[ent.label_]) # random entity of same label
        query = query.replace(ent.text, replacement)

    parsed = nlp(query.lower())

    # 1. Replace nouns and adjectives w/ antonyms from WordNet
    for token in parsed:
        orig_query.append(token.text)
        antonyms = []
        if token.pos_ not in ('ADJ', 'NOUN', 'INTJ'):
            attacked_query.append(token.text)
        elif args.aug_method == 'addsent-strong' and (classification == "REFUTES" or flipped == False):
            attacked_query.append(token.text)
        else:   # antonym replacement
            synsets = wn.synsets(token.text)
            for synset in synsets:
                for l in synset.lemmas():
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name())
            if not antonyms:
                attacked_query.append(token.text)
            else:
                attacked_query.append(' '.join(antonyms[0].split('_')))
    
    # 2. Change named entities and numbers to nearest word in GloVe word vector space
    ch_query = []
    ents_lst = [e.text for e in parsed.ents]
    for token in attacked_query:
        if token in ents_lst and token in nearby_word_dict and token != "place":
            # replace with closest distance word 
            i = 1
            nearest_list = nearby_word_dict[token]
            nearest = nearest_list[i]["word"]
            while editdistance.eval(nearest, token) < 3 and i < len(nearest_list) - 1:
                i += 1
                nearest = nearby_word_dict[token][i]["word"]
            ch_query.append(nearest)
        else:
            ch_query.append(token)

    ch_query = ' '.join(ch_query)

    # 3. Did the query change? 
    if ch_query.lower().replace(" ", "") == query.lower().replace(" ", ""): 
        ch_query = replace_nearby_words(ch_query)

    assert(o_query != ch_query) # assert should never fail
    return ch_query


def augment(dataset_name, aug_method, k, sentences, objs, client, insert_pos=None):
    AUG_MAX_POS_FOR_MANY = {
        'movies': 20,
        'fever': 9,
        'multirc': 11,
        'imdb': 6,
        'boolq_truncated': 15,
    }
    
    AUG_MAX_POS = {
        'fever': 9,
        'multirc': 13,
    }

    for obj in objs:
        obj.augs = []
    all_tokens = [tok for sent in sentences for tok in sent.split()]

    if aug_method.startswith('random_sent'):
        if aug_method == 'random_sent':
            random_sentences = load_random_sentences(k)

        elif aug_method == 'random_sent_wt103':
            random_sentences = load_random_sentences_from_wikitext103(k)
        
        for random_sentence in random_sentences:
            num_sents = len(sentences)
            if insert_pos is None:
                insert_pos = random.randint(0, min(num_sents, AUG_MAX_POS_FOR_MANY[dataset_name]))
            sent_lens = [len(s.strip().split()) for s in sentences]
            accum = np.cumsum(sent_lens).tolist()

            num_before_insert_tokens = len([tok for sent in sentences[:insert_pos] for tok in sent.split()])
            token_shift = len(random_sentence.split())
            sentence_shift = 1
            for obj in objs:
                for i in range(len(obj.evidences)):
                    if obj.evidences[i][0]['start_token'] >= num_before_insert_tokens:
                        obj.evidences[i][0]['start_token'] += token_shift
                        obj.evidences[i][0]['end_token'] += token_shift
                        obj.evidences[i][0]['start_sentence'] += sentence_shift
                        obj.evidences[i][0]['end_sentence'] += sentence_shift
                # shift old aug positions if inserted new pos before them
                obj.augs = [p if p < insert_pos else p + 1 for p in obj.augs]
                obj.augs.append(insert_pos)
            sentences.insert(insert_pos, random_sentence)
        return sentences, objs

    elif aug_method == 'addsent' or aug_method == 'addsent-strong':
        file_names = []
        docs = []

        for obj in objs:
            if dataset_name == 'movies':
                file_name = obj.annotation_id
            elif dataset_name == 'multirc':
                file_name = obj.annotation_id
                obj.docids = [file_name]
            else:
                file_name = str(obj.annotation_id) + '+' + obj.docids[0]
                obj.docids[0] = file_name
            file_names.append(file_name)
            if dataset_name == 'fever':
                aug = addsent_attack(obj.query, obj.classification)
            elif dataset_name == 'multirc':
                split = addsent_multirc(obj.query).split("||", maxsplit=1)
                question = split[0]
                answer = split[1]
                response = client.query_const_parse(question, add_ner=True)
                aug, miss = convert_queries.convert(question, answer, response['sentences'][0])
            if aug is not None:  # there is an attack (in some cases no attack is performed)
                num_sents = len(sentences)
                insert_pos = min(num_sents, insert_pos, AUG_MAX_POS[dataset_name])
                sent_lens = [len(s.strip().split()) for s in sentences]
                accum = np.cumsum(sent_lens).tolist()
                num_before_insert_tokens = len([tok for sent in sentences[:insert_pos] for tok in sent.split()])
                token_shift = len(aug.split())
                sentence_shift = 1

                for i in range(len(obj.evidences)):
                    for j in range(len(obj.evidences[i])):
                        if obj.evidences[i][j]['start_token'] >= num_before_insert_tokens:
                            obj.evidences[i][j]['start_token'] += token_shift
                            obj.evidences[i][j]['end_token'] += token_shift
                            obj.evidences[i][j]['start_sentence'] += sentence_shift
                            obj.evidences[i][j]['end_sentence'] += sentence_shift
                        obj.evidences[i][j]['docid'] = file_name
                docs.append(sentences[:insert_pos] + [aug] + sentences[insert_pos:])
                obj.augs.append(insert_pos)
        return docs, objs, file_names


def aug_and_cache(dataset_name, attack_dir, aug_method, k, client, insert_pos):
    DOCS_DIR = config.DOCS_DIR.format(dataset_name)
    DEV_DATA_PATH = config.DEV_DATA_PATH.format(dataset_name)
    MODIFIED_DEV_DATA_PATH = config.MODIFIED_DEV_DATA_PATH.format(dataset_name, attack_dir)
    AUG_DATA_DIR = config.AUG_DATA_DIR.format(dataset_name, attack_dir)
    Path(AUG_DATA_DIR).mkdir(parents=True, exist_ok=True)

    docid_to_examples = defaultdict(list)
    with open(DEV_DATA_PATH) as f:
        for line in f:
            obj = json.loads(line)
            obj = Namespace(**obj)
            docid = obj.evidences[0][0]['docid']
            docid_to_examples[docid].append(obj)
    os.chdir(DOCS_DIR)
    modified_dev_data = []
    num_files = len(os.listdir('.'))

    if aug_method == 'addsent' or aug_method == 'addsent-strong':
        for file_name in os.listdir('.'):
            if file_name in docid_to_examples:
                objs = docid_to_examples[file_name]
                for obj in objs:
                    extract_ents(obj.query)

    for file_name in tqdm(os.listdir('.'), total=num_files):
        sentences = []
        with open(os.path.join(DOCS_DIR, file_name)) as f:
            for line in f:
                sentences.append(line.strip())

        if aug_method == 'scramble':
            if file_name in docid_to_examples:
                random.shuffle(sentences)
                objs = docid_to_examples[file_name]
                modified_dev_data += objs
                with open(os.path.join(AUG_DATA_DIR, file_name), 'w') as f:
                    for line in sentences:
                        f.write(line + '\n')

        elif aug_method == 'addsent' or aug_method == 'addsent-strong':
            if file_name in docid_to_examples:
                objs = docid_to_examples[file_name]
                new_docs, objs, new_file_names = augment(dataset_name, aug_method, k, sentences, objs, client, insert_pos)
                modified_dev_data += objs
                for new_file_name, sentences in zip(new_file_names, new_docs):
                    with open(os.path.join(AUG_DATA_DIR, new_file_name), 'w') as f:
                        for line in sentences:
                            f.write(line + '\n')
        else:
            if file_name in docid_to_examples:
                objs = docid_to_examples[file_name]
                sentences, objs = augment(dataset_name, aug_method, k, sentences, objs, client, insert_pos)
                modified_dev_data += objs

            with open(os.path.join(AUG_DATA_DIR, file_name), 'w') as f:
                for line in sentences:
                    f.write(line + '\n')
    print(f'Augmented input data (sentences) saved under: {AUG_DATA_DIR}')
   
    Path(AUG_DATA_DIR).mkdir(parents=True, exist_ok=True)
    with open(MODIFIED_DEV_DATA_PATH, 'w') as f:
        for obj in modified_dev_data:
            obj_str = json.dumps(vars(obj))
            f.write(obj_str + '\n')
        print(f'Modified dev data saved at: {MODIFIED_DEV_DATA_PATH}')


if __name__ == '__main__':
    """
    Run:
    Add Random Sentence Attack:
        python -m rr.attacks.aug_and_cache_data --dataset-name fever --aug-method random_sent --k 1 --attack-dir addrand_pos0 --insert_pos 0

    Add Random Sentence Attack:
        python -m rr.attacks.aug_and_cache_data --dataset-name fever --aug-method random_sent_wt103 --k 1 --attack-dir addwiki_pos0 --insert_pos 0

    AddOneSent Attack:
        python -m rr.attacks.aug_and_cache_data --dataset-name fever --aug-method addsent --attack-dir addsent --insert_pos 0

    Scramble Attack:
        python -m rr.attacks.aug_and_cache_data --dataset-name fever --aug-method scramble --attack-dir scramble

    For FEVER (stronger attack):
        python -m rr.attacks.aug_and_cache_data --dataset-name fever --attack-dir addsent_attacks_strong --aug-method addsent-strong

    For FEVER (default attack):
        python -m rr.attacks.aug_and_cache_data --dataset-name fever --attack-dir addsent_attacks_default --aug-method addsent
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True, help="[movies | fever | multirc | imdb]")
    parser.add_argument("--attack-dir", type=str, required=True, help="Will create attack dir if not exist.")
    parser.add_argument("--aug-method", type=str, required=True, help="[add_one_to_top | random_sent | random_sent_wt103 | addsent | addsent-strong | scramble]")
    parser.add_argument("--k", type=int, default=1, help="Number of sentences to inject.")
    parser.add_argument("--insert_pos", type=int, default=None, help="Position of the inserted attack sentence.")
    args = parser.parse_args()

    with open(NEARBY_GLOVE_FILE) as json_file: 
        nearby_word_dict = json.load(json_file)   

    with corenlp.CoreNLPServer(port=CORENLP_PORT, logfile=CORENLP_LOG) as server:
        client = corenlp.CoreNLPClient(port=CORENLP_PORT)
        aug_and_cache(args.dataset_name, args.attack_dir, args.aug_method, args.k, client, args.insert_pos)
