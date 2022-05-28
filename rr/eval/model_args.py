"""
TODO: do a major cleanup
"""
import json
from pathlib import Path

from rr.config import Config

config = Config()

PREDICTION_DIR = 'your/saved/predictions/dir/'
BASE_DIR = '/your/repo/path/rr/base/explainable_qa'
CKPT_PATH_TEMPLATE = BASE_DIR + '/out/{}/best_model/pytorch_model.bin'
MODEL_PARAMS_PATH_TEMPLATE = BASE_DIR + '/params/{}'

MODEL_TO_PARAM_FILE = {
    # regular fever models
    'fever_ib_KL_1.0': 'gated_verification.json', # pi=0.4
    'fever_ib_KL_1.0_semi0.9': 'gated_verification.json', # pi=0.4
    'fever_vib_pi0.2': 'gated_verification_pi0.2.json',
    'fever_vib_pi0.2_semi0.9': 'gated_verification_pi0.2.json',
    'fever_fullcontext': 'bert_verification.json',
    'fever_sent_multitask': 'bert_verification.json',

    # sentences shuffled for training
    'fever_shuffle_vib_pi0.4': 'gated_verification.json',
    'fever_shuffle_vib_pi0.4_semi0.9': 'gated_verification.json',
    'fever_shuffle_sent_fullcontext': 'bert_verification.json',
    'fever_shuffle_sent_multitask': 'bert_verification.json',
    'fever_shuffle_vib_pi0.2': 'gated_verification_pi0.2.json',
    'fever_shuffle_vib_pi0.2_semi0.9': 'gated_verification_pi0.2.json',

    # fever kl tuning
    'fever_vib_pi0.4_beta0.0': 'gated_verification/pi0.4_beta0.0.json',
    'fever_vib_pi0.4_beta1.0': 'gated_verification/pi0.4_beta1.0.json',
    'fever_vib_pi0.4_beta2.0': 'gated_verification/pi0.4_beta2.0.json',
    'fever_vib_pi0.4_semi0.9_beta0.0': 'gated_verification/pi0.4_beta0.0.json',
    'fever_vib_pi0.2_semi0.9_beta0.0': 'gated_verification/pi0.2_beta0.0.json',
    'fever_vib_pi0.4_semi0.9_beta0.1': 'gated_verification/pi0.4_beta0.1.json',
    'fever_vib_pi0.2_semi0.9_beta0.1': 'gated_verification/pi0.2_beta0.1.json',
    'fever_vib_pi0.4_semi0.9_beta1.0': 'gated_verification/pi0.4_beta1.0.json',
    'fever_vib_pi0.2_semi0.9_beta1.0': 'gated_verification/pi0.2_beta1.0.json',
    'fever_vib_pi0.4_semi0.9_beta2.0': 'gated_verification/pi0.4_beta2.0.json',
    'fever_vib_pi0.2_semi0.9_beta2.0': 'gated_verification/pi0.2_beta2.0.json',
    'fever_vib_pi0.4_semi0.9_beta5.0': 'gated_verification/pi0.4_beta5.0.json',
    'fever_vib_pi0.2_semi0.9_beta5.0': 'gated_verification/pi0.2_beta5.0.json',
    'fever_vib_pi0.4_semi0.9_beta1.0_gamma0.5': 'gated_verification/pi0.4_beta1.0_gamma0.5.json',
    'fever_vib_pi0.4_semi0.9_beta1.0_gamma2.0': 'gated_verification/pi0.4_beta1.0_gamma2.0.json',
    'fever_multitask_gamma1.0': 'bert_verification.json',
    'fever_multitask_gamma2.0': 'bert_verification.json',

    # regular multirc models
    'multirc_vib': 'gated_truefalse.json',
    'multirc_vib_pi0.25': 'gated_truefalse_pi0.25.json',
    'multirc_vib_semi': 'gated_truefalse.json',
    'multirc_vib_semi_pi0.25': 'gated_truefalse_pi0.25.json',
    'multirc_fullcontext': 'bert_truefalse.json',
    'multirc_multitask': 'bert_truefalse.json',

    # multirc kl tuning
    'multirc_vib_pi0.25_beta0.0': 'gated_truefalse/pi0.25_beta0.0.json',
    'multirc_vib_pi0.25_beta0.05': 'gated_truefalse/pi0.25_beta0.05.json',
    'multirc_vib_pi0.25_beta0.1': 'gated_truefalse/pi0.25_beta0.1.json',
    'multirc_vib_pi0.25_beta0.5': 'gated_truefalse/pi0.25_beta0.5.json',
    'multirc_vib_pi0.25_beta1.0': 'gated_truefalse/pi0.25_beta1.0.json',
    'multirc_vib_pi0.25_semi0.9_beta0.0': 'gated_truefalse/pi0.25_beta0.0.json',
    'multirc_vib_pi0.25_semi0.9_beta0.1': 'gated_truefalse/pi0.25_beta0.1.json',
    'multirc_vib_pi0.25_semi0.9_beta0.05': 'gated_truefalse/pi0.25_beta0.05.json',
    'multirc_vib_pi0.25_semi0.9_beta1.0': 'gated_truefalse/pi0.25_beta1.0.json',
    'multirc_multitask_gamma1.0': 'bert_truefalse.json',
    'multirc_multitask_gamma2.0': 'bert_truefalse.json',

    # multirc sentences shuffled for training
    'multirc_shuffle_vib_pi0.25': 'gated_truefalse.json',
    'multirc_shuffle_vib_pi0.25_semi0.9': 'gated_truefalse.json',
    'multirc_shuffle_sent_fullcontext': 'bert_truefalse.json',
    'multirc_shuffle_sent_multitask': 'bert_truefalse.json',

    # regular movies models
    'movies_KL_1.0-1': 'gated_sentiment.json',
    'movies_KL_1.0_semi0.9': 'gated_sentiment.json',
    'movies_fullcontext': 'bert_sentiment.json',

    # regular imdb models
    'imdb_vib': 'gated_sentiment.json',
    'imdb_fullcontext': 'bert_sentiment.json',

    'try': 'gated_verification.json',
}

class Args:
    def __init__(self,
                 dataset_name,
                 bottleneck_type,
                 intervene=None,
                 data_dir=None,
                 attack_dir=None,
                 model_name=None,
                 output_dir_name=None,
                 output_to_tmp=False
        ):
        """
        dataset_name = [movies | fever | multirc]
        """
        if model_name is None:
            self.model_ckpt_path = None
            self.model_params_path = None
        else:
            self.model_ckpt_path = CKPT_PATH_TEMPLATE.format(model_name)
            self.model_params_path = MODEL_PARAMS_PATH_TEMPLATE.format(MODEL_TO_PARAM_FILE[model_name])

        if dataset_name in ('movies', 'rand_sent'):
            max_seq_length = 512
            max_query_length = 4
            max_num_sentences = 36
            if bottleneck_type == 'vib':
                model_type = 'distilbert_gated_sent'
            elif bottleneck_type == 'vib_semi':
                model_type = 'distilbert_gated_sent'
            elif bottleneck_type == 'full':
                model_type = 'bert'
                self.predicted_eval_evidence_file = None
                self.random_evidence = False
                self.focus_attention = False
                self.pal_attention = False
                self.print_example = False

        elif dataset_name == 'imdb':
            max_seq_length = 512
            max_query_length = 4
            max_num_sentences = 36
            if bottleneck_type == 'vib':
                model_type = 'distilbert_gated_sent'
            elif bottleneck_type == 'full':
                model_type = 'bert'
                self.predicted_eval_evidence_file = None
                self.random_evidence = False
                self.focus_attention = False
                self.pal_attention = False
                self.print_example = False

        elif dataset_name == 'multirc':
            max_seq_length = 512
            max_query_length = 32
            max_num_sentences = 15

            if bottleneck_type == 'vib':
                model_type = 'distilbert_hard_gated_sent'
            elif bottleneck_type == 'vib_semi':
                model_type = 'distilbert_hard_gated_sent'
            elif bottleneck_type == 'full':
                model_type = 'bert'
                self.predicted_eval_evidence_file = None
                self.random_evidence = False
                self.focus_attention = False
                self.pal_attention = False
                self.print_example = False
            elif bottleneck_type == 'full_multitask':
                model_type = 'multitask_bert'
                self.out_domain = False
                self.predicted_eval_evidence_file = None
                self.random_evidence = False
                self.print_example = False
        elif dataset_name == 'fever':
            max_seq_length = 512
            max_query_length = 32
            max_num_sentences = 10

            if bottleneck_type == 'vib':
                model_type = 'distilbert_gated_sent'
            elif bottleneck_type == 'vib_semi':
                model_type = 'distilbert_gated_sent'
            elif bottleneck_type == 'full':
                model_type = 'bert'
                self.predicted_eval_evidence_file = None
                self.random_evidence = False
                self.focus_attention = False
                self.pal_attention = False
                self.print_example = False
            elif bottleneck_type == 'full_multitask':
                model_type = 'multitask_bert'
                self.out_domain = False
                self.predicted_eval_evidence_file = None
                self.random_evidence = False
                self.print_example = False
        elif dataset_name == 'boolq':
            max_seq_length = 512
            max_query_length = 24
            max_num_sentences = 25
            if bottleneck_type == 'vib':
                model_type = 'distilbert_gated_sent'
            elif bottleneck_type == 'vib_semi':
                model_type = 'distilbert_gated_sent'
            elif bottleneck_type == 'full':
                model_type = 'bert'
                self.predicted_eval_evidence_file = None
                self.random_evidence = False
                self.focus_attention = False
                self.pal_attention = False
                self.print_example = False
            elif bottleneck_type == 'full_multitask':
                model_type = 'multitask_bert'
                self.out_domain = False
                self.predicted_eval_evidence_file = None
                self.random_evidence = False
                self.print_example = False

        if output_dir_name is None:
            output_dir_name = dataset_name

        if intervene is None:
            # no intervention
            output_dir = f'{PREDICTION_DIR}/{output_dir_name}/{bottleneck_type}/{model_name}/original/'
        elif intervene == 'attack':
            assert attack_dir is not None
            output_dir = f'{PREDICTION_DIR}/{output_dir_name}/{bottleneck_type}/{model_name}/{attack_dir}/'
        else:
            raise ValueError('Unsupported intervening option.')

        # override if `output_to_tmp` is set
        output_dir = output_dir if not output_to_tmp else f'{PREDICTION_DIR}/tmp/'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.bottleneck_type = bottleneck_type
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.intervene = intervene
        self.local_rank = -1
        if data_dir is None:
            self.data_dir = f'{BASE_DIR}/data/{dataset_name}'
        else:
            self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.gold_evidence = False
        self.overwrite_cache = False
        self.truncate = False
        self.debug = False
        self.low_resource = False
        self.model_type = model_type
        self.max_num_sentences = max_num_sentences
        self.eval_batch_size = 1
        self.per_gpu_eval_batch_size = 1
        self.n_gpu = 1
        self.num_evaluations = 1
        self.device = 'cuda'
        self.warm_start = False
        self.semi_supervised = 0.0

        self.intervention_map = {}

        if self.model_params_path is None:
            self.model_params = {}
        else:
            with open(self.model_params_path) as f:
                self.model_params = json.load(f)

        self.model_params['max_query_length'] = self.max_query_length
        self.model_params['warm_start'] = self.warm_start
        self.model_params['semi_supervised'] = self.semi_supervised
