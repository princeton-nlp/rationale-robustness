import re
import random
import string

from tqdm import tqdm
import pandas as pd
from nltk.corpus import words

from rrtl.dataloaders import extract_token_rationales
from rrtl.config import Config

config = Config()

random.seed(0)


def load_data(dataset_name, mode, k_p=2, k_h=2, neg_source='wn'):
    if neg_source == 'wn':
        word_pool = get_wordnet_pool()
    elif neg_source == 'wt103':
        word_pool = get_wt103_pool()

    df = pd.read_csv(config.DATA_DIR / dataset_name / f'{mode}.csv')
    df = df.dropna()

    data = []
    df_list = list(df.iterrows())
    rand_word_index = 0
    for index, row in tqdm(df_list, total=len(df_list)):
        label = row['gold_label'].strip()
        highlight_premise = row['Sentence1_marked_1'].strip()
        highlight_hypothesis = row['Sentence2_marked_1'].strip()

        # sampling random words (neg rationales)
        p_rand_words = word_pool[rand_word_index:rand_word_index + k_p]
        h_rand_words = word_pool[rand_word_index + k_p:rand_word_index + k_p + k_h]
        rand_word_index += (len(p_rand_words) + len(h_rand_words))
        rand_word_index = rand_word_index % len(word_pool)

        premise, highlight_premise = inject_random_words(row['Sentence1'].strip(), highlight_premise, p_rand_words)
        hypothesis, highlight_hypothesis = inject_random_words(row['Sentence2'].strip(), highlight_hypothesis, h_rand_words)
    
        premise_spans = extract_token_rationales(premise, highlight_premise)
        hypothesis_spans = extract_token_rationales(hypothesis, highlight_hypothesis)

        neg_premise_spans = extract_token_rationales(premise, highlight_premise, highlight_mode='neg')
        neg_hypothesis_spans = extract_token_rationales(hypothesis, highlight_hypothesis, highlight_mode='neg')

        data.append({
            'label': label,
            'premise': premise,
            'hypothesis': hypothesis,
            'highlight_premise': highlight_premise,
            'highlight_hypothesis': highlight_hypothesis,
            'premise_spans': premise_spans,
            'hypothesis_spans': hypothesis_spans,
            'neg_premise_spans': neg_premise_spans,
            'neg_hypothesis_spans': neg_hypothesis_spans,
        })
    return data


def load_cached_data(path):
    df = pd.read_csv(path, sep='\t')
    print(df)
    return df


def inject_random_words(text, highlight_text, rand_words):
    tokens = text.split()
    highlight_tokens = highlight_text.split()
    if len(tokens) >= len(rand_words):
        inject_inds = random.sample(range(len(tokens)), len(rand_words))
    else:
        inject_inds = list(range(len(rand_words)))

    for inject_ind, rand_word in zip(inject_inds, rand_words):
        tokens.insert(inject_ind, rand_word)
        highlight_tokens.insert(inject_ind, '[' + rand_word + ']')
    text = ' '.join(tokens)
    highlight_text = ' '.join(highlight_tokens)
    return text, highlight_text


def get_wt103_pool():
    word_pool = []
    with open(config.WIKITEXT103_DEV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word_pool += line.split()
    word_pool = list(set(word_pool))
    random.shuffle(word_pool)
    return word_pool


def get_wordnet_pool():
    word_pool = words.words()
    random.shuffle(word_pool)
    return word_pool


if __name__ == '__main__':
    """
    python -m rrtl.dataset_utils.cache_neg_rationale
    """
    neg_source = 'wn'  # [wn | wt103]
    for mode in ('train', 'dev', 'test'):
        path = config.NEG_RATIONALE_DIR / f'{mode}.csv'
        data = load_data('esnli', mode)
        df = pd.DataFrame(data)
        df.to_csv(path, sep='\t')