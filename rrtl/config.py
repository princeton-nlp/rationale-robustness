from pathlib import Path


class Config:
    ROOT_DIR = Path('/path/to/your/repo/')
    PROJECT_NAME = 'rationale-robustness'

    PROJECT_DIR = ROOT_DIR / PROJECT_NAME
    EXP_DIR = PROJECT_DIR / 'experiments'
    DATA_DIR = PROJECT_DIR / 'data'
    CACHE_DIR = PROJECT_DIR / 'cache'

    TOK_KWARGS = {
        'padding': 'max_length',
        'truncation': True,
        'return_offsets_mapping': True,
        'return_tensors': 'pt',
    }

    FEVER_LABEL = {
        'REFUTES': 0,
        'SUPPORTS': 1,
    }

    MULTIRC_LABEL = {
        'False': 0,
        'True': 1,
    }

    BEER_LABEL = {
        'neg': 0,
        'pos': 1,
    }

    HOTEL_LABEL = {
        'neg': 0,
        'pos': 1,
    }

    WIKITEXT103_DEV_PATH = ROOT_DIR / 'rationale-robustness' / 'data' / 'wikitext-103' / 'wiki.valid.tokens'
    NEG_RATIONALE_DIR = ROOT_DIR / PROJECT_NAME / 'data' / 'neg_rationale'
