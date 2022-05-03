import argparse
import json
from nltk.tokenize import word_tokenize
import numpy as np
from scipy import spatial
from sklearn.neighbors import KDTree
import string
import sys
from tqdm import tqdm
import os
import re

OPTS = None

SQUAD_PATH = '/n/fs/nlp-jh70/rationale-robustness-token-level/data/squad/train-v1.1.json'
WORDVEC_PATH = '/n/fs/nlp-jh70/rationale-lff/lff/attacks/glove.6B.100d.txt'
SAVE_PATH = '/n/fs/nlp-jh70/rationale-robustness-token-level/rrtl/attacks/jh_out/'
PUNCTUATION = set(string.punctuation) | set(['``', "''"])

def parse_args():
  parser = argparse.ArgumentParser('Find nearby words for words in dataset.')
  parser.add_argument('--wordvec_file', help='File with word vectors.', default=WORDVEC_PATH)
  parser.add_argument('--dir_path', '-f',
                      help=('dir file (defaults to all squad training file).'),
                      default=SQUAD_PATH)
  parser.add_argument('--num-neighbors', '-n', type=int, default=100,
                      help='Number of neighbors per word (default = 100).')
  return parser.parse_args()

def extract_words():
    words = set()
    with open(OPTS.dir_path) as f:
        obj = json.load(f)
        for data in obj['data']:
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                cur_words = set(w.lower() for w in word_tokenize(context) if w not in PUNCTUATION)
                words |= cur_words
    return words

def get_nearby_words(main_words):
  main_inds = {}
  all_words = []
  all_vecs = []
  with open(OPTS.wordvec_file) as f:
    for i, line in tqdm(enumerate(f)):
      toks = line.rstrip().split(' ')
      word = str(toks[0])
      vec = np.array([float(x) for x in toks[1:]])
      all_words.append(word)
      all_vecs.append(vec)
      if word in main_words:
        main_inds[word] = i
  print('Found vectors for %d/%d words = %.2f%%' % (
      len(main_inds), len(main_words), 100.0 * len(main_inds) / len(main_words)))
  tree = KDTree(np.array(all_vecs))
  nearby_words = {}
  for word in tqdm(main_inds):
    dists, inds = tree.query([all_vecs[main_inds[word]]],
                             k=OPTS.num_neighbors + 1)
    nearby_words[word] = [
        {'word': all_words[i], 'dist': d} for d, i in zip(dists[0], inds[0])]
    print(nearby_words[word])
  return nearby_words

def main():
  words = extract_words()
  print('Found %d words' % len(words))
  nearby_words = get_nearby_words(words)
  with open(SAVE_PATH + 'squad-nearest100.json', 'w') as json_file:
    json.dump(nearby_words, json_file)

if __name__ == '__main__':
  OPTS = parse_args()
  main()