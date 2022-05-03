"""Reorder adv-squad attack position and save to file."""
import json
import random
from pprint import pprint
from nltk.tokenize import sent_tokenize
from nltk.corpus import words

from rrtl.config import Config

config = Config()

ADV_SQUAD_ADDONESENT_PATH = config.DATA_DIR / f'squad/adv/sample1k-HCVerifySample.json'


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


def load_adv_squad_addonesent():
    datapoints = []
    with open(ADV_SQUAD_ADDONESENT_PATH) as f:
        obj = json.load(f)
        for data in obj['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                orig_context_char_length = len(context)
                context = context.lstrip()
                num_head_spaces = orig_context_char_length - len(context)

                qas = paragraph['qas']
                for qa in qas:
                    example_id = qa['id']
                    question = qa['question'].strip()
                    answer = qa['answers'][0]
                    answer_texts = [answer['text'] for answer in qa['answers']]
                    start = answer['answer_start'] - num_head_spaces
                    end = answer['answer_start'] + len(answer['text']) - num_head_spaces

                    # TODO: refactor
                    sents = sent_tokenize(context)
                    context = ' '.join(sents)
                    limit = 0
                    while answer_texts[0] != context[start:end]:
                        start += 1
                        end += 1
                        limit += 1
                        if limit == 10:
                            break
                    assert answer_texts[0] == context[start:end]
    
                    datapoints.append({
                        'example_id': example_id,
                        'context': context,
                        'sents': sents,
                        'question': question,
                        'start': start,
                        'end': end,
                        'answer_texts': answer_texts,
                    })
    return datapoints


def reorder_attack_position(datapoints, new_position, attack_type):
    processed_datapoints = []
    for datapoint in datapoints:
        example_id = datapoint['example_id']
        if 'turk' not in example_id:
            #processed_datapoints.append(datapoint)
            continue
        
        answer_texts = datapoint['answer_texts']
        sents = datapoint['sents']
        if attack_type == 'sent':
            attack = sents[-1]
        elif attack_type == 'rand':
            attack = load_random_sentence()
        elif attack_type == 'wiki':
            attack = load_random_sentence_from_wikitext103()
        sents = sents[:-1]
        if new_position == 0:
            sents.insert(new_position, attack)
            start = datapoint['start'] + len(attack) + 1
            end = datapoint['end'] + len(attack) + 1
        elif new_position == -1:
            sents.append(attack)
            start = datapoint['start']
            end = datapoint['end']
        context = ' '.join(sents)
        if answer_texts[0] != context[start:end]:
            continue

        processed_datapoints.append({
            'example_id': example_id,
            'context': context,
            'sents': sents,
            'question': datapoint['question'],
            'start': start,
            'end': end,
            'answer_texts': answer_texts,
        })
    return processed_datapoints


def save(datapoints, save_path):
    with open(save_path, 'w') as f:
        for datapoint in datapoints:
            obj_string = json.dumps(datapoint)
            f.write(obj_string + '\n')


if __name__ == '__main__':
    """
    python -m rrtl.dataset_utils.cache_adv_squad_attack_pos
    """
    attack_type = 'sent'
#    attack_type = 'rand'
#    attack_type = 'wiki'
#    attack_pos = 0
    attack_pos = 1
    #SAVE_PATH = config.DATA_DIR / f'squad/adv/sample1k-HCVerifySample-pos0.jsonl'
    if attack_pos >= 0:
        SAVE_PATH = config.DATA_DIR / f'squad/adv/add{attack_type}_pos{attack_pos}.jsonl'
    elif attack_pos == -1:
        SAVE_PATH = config.DATA_DIR / f'squad/adv/add{attack_type}.jsonl'

    datapoints = load_adv_squad_addonesent()
    datapoints = reorder_attack_position(datapoints, new_position=attack_pos, attack_type=attack_type)
    save(datapoints, SAVE_PATH)