"""
Sample from yelp review dataset for sentiment topical attack.
- topically/lexically similar to the gold rationales in the beer dataset

python -m rrtl.attacks.sentiment_attack
"""
import random
from collections import Counter

import pandas as pd
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from nltk.corpus import words

from rrtl.config import Config

config = Config()



cache_dir = config.DATA_DIR / 'yelp'

beer_p = 0
beer_n = 0


def load_random_sentence_from_wikitext103():
    wiki_path = config.DATA_DIR / 'wikitext-103/wiki.valid.tokens'
    sentences = []
    with open(wiki_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentences += [sent.strip() for sent in line.split('.')]
    random.shuffle(sentences)
    return sentences[0]


def load_random_sentence():
    return ' '.join(random.sample(words.words(), 10))


def cache_yelp_sentences():
    dataset = load_dataset(
        'yelp_review_full',
        split='train',
        cache_dir=cache_dir
    )
    print(len(dataset))
    all_sents = []
    for datapoint in dataset:
        sents = sent_tokenize(datapoint['text'])
        all_sents += [sent.strip('\n') for sent in sents]
    
    print(len(all_sents))
    with open(cache_dir + 'yelp_sentences.txt', 'w') as f:
        for sent in all_sents:
            sent = sent.replace('\n','')
            sent = sent.replace('\t','')
            f.write(sent + '\n')


def load_attack_pool():
    attack_pool = []
    with open(cache_dir + 'yelp_sentences.txt') as f:
        for line in f:
            line = line.replace('\n','')
            line = line.replace('\t','')
            attack_pool.append(line.strip())
    return attack_pool


def attack_beer(datapoint, attack_pool, attack_pos, attack_type):
    global beer_p, beer_n
    ATTACK_TYPE = -1
    attack_dict = [
        {
            1: 'the coffee is extremely bad , the color is totally off .',
            0: 'the coffee is super awesome , it looks beautiful .'
        },
        {
            1: 'the carpet is extremely bad , the color is totally off .',
            0: 'the carpet is super awesome , it looks beautiful .'
        },
        {
            1: 'the carpet looks disgusting .',
            0: 'the carpet looks pretty .'
        },
        {
            1: 'the tea looks bad .',
            0: 'the tea looks good .'
        },
        {
            1: 'the tea looks disgusting .',
            0: 'the tea looks beautiful .'
        },
        {
            1: 'the tea looks strange .',
            0: 'the tea looks fancy .'
        },
        {
            1: 'the tea looks horrible .',
            0: 'the tea looks fabulous .'
        },
        {
            1: 'the tea looks bad .',
            0: 'the tea looks good .'
        },
        {
            1: 'the tea looks strange .',
            0: 'the tea looks fine .'
        },
        {
            1: 'the carpet looks bad .',
            0: 'the carpet looks good .'
        },
        {
            1: 'the carpet looks really ugly .',
            0: 'the carpet looks very beautiful .'
        },
    ]
    if attack_type is None:
        if datapoint['label'] > 0.5:
            attack = attack_dict[ATTACK_TYPE][1]
            beer_p += 1
        elif datapoint['label'] <= 0.5:
            attack = attack_dict[ATTACK_TYPE][0]
            beer_n += 1
    elif attack_type == 'addrand':
        attack = load_random_sentence()
    elif attack_type == 'addwiki':
        attack = load_random_sentence_from_wikitext103()

    sents = sent_tokenize(datapoint['text'])
    if attack_pos == 0:
        sents = [attack] + sents
        rationale = [int(r) for r in datapoint['rationale'].split()]
        rationale = [-1] * len(attack.split()) + rationale
    elif attack_pos == -1:
        sents = sents + [attack]
        rationale = [int(r) for r in datapoint['rationale'].split()]
        rationale = rationale + [-1] * len(attack.split())

    datapoint['text'] = ' '.join(sents)
    datapoint['rationale'] = ' '.join([str(r) for r in rationale])
    assert len(datapoint['text'].split()) == len(rationale)
    return datapoint


def attack_hotel(datapoint, attack_pool, attack_pos, attack_type):
    ATTACK_TYPE = -1
    attack_dict = [
        {
            1: 'My car was filthy and it has moss all over it .',
            0: 'My car is very clean, shiny, and totally spotless .'
        },
        {
            1: 'My car was not washed and it has moss all over it .',
            0: 'My car is very new, shiny, and totally spotless .'
        },
        {
            1: 'My car was green and it has moss all over it .',
            0: 'My car is very cool, stylish, and totally eye-cathcing .'
        },
        {
            1: 'My car is very wretched .', 
            0: 'My car is very safe .' 
        },
        {
            1: 'My car is very dreary .', 
            0: 'My car is very transparent .'
        },
        {
            1: 'My car is very bad .',
            0: 'My car is very good .'
        },
        {
            1: 'My car is very old .',
            0: 'My car is very new .'
        },
        {
            1: 'My car is very filthy .',
            0: 'My car is very clean .'
        },
    ]
    if attack_type is None:
        attack = attack_dict[ATTACK_TYPE][datapoint['label']]
    elif attack_type == 'addrand':
        attack = load_random_sentence()
    elif attack_type == 'addwiki':
        attack = load_random_sentence_from_wikitext103()

    sents = sent_tokenize(datapoint['text'])

    if attack_pos == 0:
        sents = [attack] + sents
        rationale = [int(r) for r in datapoint['rationale'].split()]
        rationale = [-1] * len(attack.split()) + rationale
    elif attack_pos == -1:
        sents = sents + [attack]
        rationale = [int(r) for r in datapoint['rationale'].split()]
        rationale = rationale + [-1] * len(attack.split())
    datapoint['text'] = ' '.join(sents)
    datapoint['rationale'] = ' '.join([str(r) for r in rationale])
    return datapoint
    

def load_beer_dataset(mode):
    if mode == 'dev':
        path = config.DATA_DIR / 'sentiment/data/source/beer0.dev'
    elif mode == 'test':
        path = config.DATA_DIR / 'sentiment/beeradvocate/beer0.test'
    df = pd.read_csv(path, delimiter='\t')
    return [row for index, row in df.iterrows()]


def load_hotel_dataset(mode):
    if mode == 'dev':
        path = config.DATA_DIR / f'sentiment/data/oracle/hotel_Cleanliness.dev'
    elif mode == 'test':
        path = config.DATA_DIR / f'sentiment/data/target/hotel_Cleanliness.train'
    df = pd.read_csv(path, delimiter='\t')
    return [row for index, row in df.iterrows()]


def attack_and_save_beer(mode, attack_pos, attack_type):
    attacked_datapoints = []
    beer_datapoints = load_beer_dataset(mode)
    attack_pool = load_attack_pool()
    
    for datapoint in beer_datapoints:
        attacked_datapoint = attack_beer(datapoint, attack_pool=attack_pool, attack_pos=attack_pos, attack_type=attack_type)
        attacked_datapoints.append(attacked_datapoint)
        
    df = pd.DataFrame(attacked_datapoints)
    if mode == 'dev':
        if attack_type is None:
            save_path = config.DATA_DIR / f'sentiment/adv/beer/beer0.dev.{attack_pos}'
        else:
            save_path = config.DATA_DIR / f'sentiment/adv/beer/beer0.dev.{attack_pos}.{attack_type}'
    elif mode == 'test':
        if attack_type is None:
            save_path = config.DATA_DIR / f'sentiment/adv/beer/beer0.test.{attack_pos}'
        else:
            save_path = config.DATA_DIR / f'sentiment/adv/beer/beer0.test.{attack_pos}.{attack_type}'

    df.to_csv(save_path, index=False, sep="\t")
    print('Data saved at:', save_path)


def attack_and_save_hotel(mode, attack_pos, attack_type):
    attacked_datapoints = []
    hotel_datapoints = load_hotel_dataset(mode)
    
    attack_pool = load_attack_pool()
    
    for datapoint in hotel_datapoints:
        attacked_datapoint = attack_hotel(datapoint, attack_pool=attack_pool, attack_pos=attack_pos, attack_type=attack_type)
        attacked_datapoints.append(attacked_datapoint)
    random.shuffle(attacked_datapoints)
    df = pd.DataFrame(attacked_datapoints)
    if mode == 'dev':
        if attack_type is None:
            save_path = config.DATA_DIR / f'sentiment/adv/hotel/hotel_Cleanliness.dev.{attack_pos}.csv'
        else:
            save_path = config.DATA_DIR / f'sentiment/adv/hotel/hotel_Cleanliness.dev.{attack_pos}.{attack_type}.csv'
    elif mode == 'test':
        if attack_type is None:
            save_path = config.DATA_DIR / f'sentiment/adv/hotel/hotel_Cleanliness.test.{attack_pos}.csv'
        else:
            save_path = config.DATA_DIR / f'sentiment/adv/hotel/hotel_Cleanliness.test.{attack_pos}.{attack_type}.csv'
    df.to_csv(save_path, index=False, sep="\t")
    print('Data saved at:', save_path)


if __name__ == '__main__':
    attack_type = None # options: ['addrand' | 'addwiki']
    attack_pos = 0
    mode = 'dev' # options: ['dev' | 'test']
    attack_and_save_hotel(mode, attack_pos, attack_type)
    attack_and_save_beer(mode, attack_pos, attack_type)