from pathlib import Path


class Config:
    # run
    seed = 0
    PROJECT_NAME = 'rationale-robustness'

    # model
    BASE_BERT_DIM = 768

    # paths
    ROOT_DIR = Path('/n/fs/nlp-hc22/rationale-robustness')
#    ROOT_DIR = Path('/path/to/your/repo')
    PROJECT_DIR = ROOT_DIR / PROJECT_NAME
    EXP_DIR = PROJECT_DIR / 'experiments'
    TMP_DIR = PROJECT_DIR / 'tmp'

    # ERASER benchmark datasets
#    DATA_DIR = PROJECT_DIR / 'data'
    DATA_DIR = PROJECT_DIR / 'rr/base/explainable_qa/data'
    
    EI = DATA_DIR / 'evidence_inference'
    BOOLQ = DATA_DIR / 'boolq'
    MOVIES = DATA_DIR / 'movies'
    FEVER = DATA_DIR / 'fever'
    MULTIRC = DATA_DIR / 'multirc'
    COSE = DATA_DIR / 'cose'
    ESNLI = DATA_DIR / 'esnli'
    SCIFACT = DATA_DIR / 'scifact'
    IMDB = DATA_DIR / 'imdb'

    eraser_dirs = {
        'movies': MOVIES,
        'fever': FEVER,
        'multirc': MULTIRC,
        'imdb': IMDB,
    }

    target_vocab = {
        'movies': {'POS': 1, 'NEG': 0},
        'fever': {'SUPPORTS': 1, 'REFUTES': 0},
        'multirc': {'True': 0, 'False': 1},
        'imdb': {'POS': 1, 'NEG': 0},
    }

    WIKITEXT103_DEV_PATH = PROJECT_DIR / 'data' / 'wikitext-103' / 'wiki.valid.tokens'


    # paths for adversarial attack
    ATTACK_DIR = str(PROJECT_DIR / 'rr/attacks/data/{}/{}')
    DOCS_DIR = str(PROJECT_DIR / 'rr/base/explainable_qa/data/{}/docs/')
    TRAIN_DATA_PATH = str(PROJECT_DIR / 'rr/base/explainable_qa/data/{}/train.jsonl')
    DEV_DATA_PATH = str(PROJECT_DIR / 'rr/base/explainable_qa/data/{}/val.jsonl')
    TEST_DATA_PATH = str(PROJECT_DIR / 'rr/base/explainable_qa/data/{}/test.jsonl')
    MODIFIED_DEV_DATA_PATH = str(PROJECT_DIR / 'rr/attacks/data/{}/{}/val.jsonl')
    AUG_DATA_DIR = str(PROJECT_DIR / 'rr/attacks/data/{}/{}/docs/')

    SHUFFLE_DOCS_DIR = str(PROJECT_DIR / 'rr/shuffle/data/{}/{}/docs/')
    SHUFFLE_TRAIN_DATA_PATH = str(PROJECT_DIR / 'rr/shuffle/data/{}/{}/train.jsonl')
    SHUFFLE_DEV_DATA_PATH = str(PROJECT_DIR / 'rr/shuffle/data/{}/{}/val.jsonl')
    SHUFFLE_TEST_DATA_PATH = str(PROJECT_DIR / 'rr/shuffle/data/{}/{}/test.jsonl')
