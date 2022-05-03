import os
import yaml
import glob
from argparse import Namespace

import torch
from torch.optim import AdamW, Adam
from rrtl.dataloaders import (
    SQUADDataLoader,
    FeverDataLoader,
    MultiRCDataLoader,
    SQUADNegRationaleDataLoader,
    SentimentDataLoader,
)
from rrtl.models.squad_models import (
    FullContextSQUADModel,
    VIBSQUADSentModel,
    VIBSQUADTokenModel,
    SpectraSQUADModel,
    SpectraSQUADTokenModel,
)
from rrtl.models.eraser_models import (
    FullContextERASERModel,
    FullContextSentimentModel,
    VIBERASERTokenModel,
    VIBERASERSentModel,
    SpectraERASERSentModel,
    VIBSentimentTokenModel,
    SpectraSentimentTokenModel,
)
from rrtl.config import Config

config = Config()


def update_args(args):
    """
    Update args to handle version differences when loading models.
    E.g., older models might miss some arguments.
    """
    if not hasattr(args, 'optimizer'):
        setattr(args, 'optimizer', 'adamw')
    if not hasattr(args, 'dataset_split'):
        setattr(args, 'dataset_split', 'all')
    if not hasattr(args, 'use_gold_rationale'):
        setattr(args, 'use_gold_rationale', False)
    if not hasattr(args, 'use_neg_rationale'):
        setattr(args, 'use_neg_rationale', False)
    if not hasattr(args, 'gamma'):
        setattr(args, 'gamma', 1.0)
    if not hasattr(args, 'dropout_rate'):
        setattr(args, 'dropout_rate', 0.1)
    if not hasattr(args, 'fix_input'):
        setattr(args, 'fix_input', None)
    if not hasattr(args, 'mask_scoring_func'):
        setattr(args, 'mask_scoring_func', 'linear')
    if not hasattr(args, 'flexible_prior'):
        setattr(args, 'flexible_prior', None)
    if not hasattr(args, 'budget_ratio'):
        setattr(args, 'budget_ratio', None)
    if not hasattr(args, 'cache_dir'):
        setattr(args, 'cache_dir', config.CACHE_DIR)
    return args


def args_factory(args):
    # model specific
    if args.encoder_type == 'bert-base-uncased':
        args.encoder_hidden_size = 768
    elif args.encoder_type == 'distilbert-base-uncased':
        args.encoder_hidden_size = 768
    elif args.encoder_type == 'roberta-large':
        args.encoder_hidden_size = 1024

	# path specific
    args.file_name = 'model-step={step}-acc={best_score:.2f}.pt'
    args.ckpt_dir = os.path.join(config.EXP_DIR, args.run_name)
    args.file_path = os.path.join(args.ckpt_dir, args.file_name)
    args.args_path = os.path.join(args.ckpt_dir, 'args.yaml')
    args.ckpt = not args.disable_ckpt
    args.use_cuda = not args.disable_cuda
    if args.debug:
        args.ckpt = False
        args.disable_ckpt = True

    # tracking 
    args.best_score = float('-inf')
    args.total_seen = 0
    args.global_step = 0

    return args


def get_model_class(args):
    if args.model_type == 'vib_squad_sent':
        model_class = VIBSQUADSentModel
    elif args.model_type == 'vib_squad_token':
        model_class = VIBSQUADTokenModel
    elif args.model_type == 'fc_squad':
        model_class = FullContextSQUADModel
    elif args.model_type == 'fc_fever' or args.model_type == 'fc_multirc':
        model_class = FullContextERASERModel
    elif args.model_type == 'vib_fever' or args.model_type == 'vib_multirc':
        model_class = VIBERASERSentModel
    elif args.model_type == 'vib_fever_token' or args.model_type == 'vib_multirc_token':
        model_class = VIBERASERTokenModel
    elif args.model_type == 'spectra':
        model_class = SpectraSQUADModel
    elif args.model_type == 'spectra_squad_token':
        model_class = SpectraSQUADTokenModel
    elif args.model_type == 'spectra_fever_sent' or args.model_type == 'spectra_multirc_sent':
        model_class = SpectraERASERSentModel
    elif args.model_type in ('fc_beer', 'fc_hotel'):
        model_class = FullContextSentimentModel
    elif args.model_type in ('vib_beer_token', 'vib_hotel_token'):
        model_class = VIBSentimentTokenModel
    elif args.model_type in ('spectra_beer_token', 'spectra_hotel_token'):
        model_class = SpectraSentimentTokenModel
    else:
        raise ValueError('Model type not implemented.')
    return model_class


def get_optimizer_class(args):
    if args.optimizer == 'adamw':
        optimizer_class = AdamW
    elif args.optimizer == 'adam':
        optimizer_class = Adam
    else:
        raise ValueError('Optimizer type not implemented.')
    return optimizer_class


def get_dataloader_class(args):
    if args.dataset_name.startswith('squad'):#, 'squad-addonesent', 'squad-addsent', 'squad-addonesent-pos0'):
        dataloader_class = SQUADDataLoader
    elif args.dataset_name == 'fever':
        dataloader_class = FeverDataLoader
    elif args.dataset_name == 'multirc':
        dataloader_class = MultiRCDataLoader
    elif args.dataset_name == 'squad-nr':
        dataloader_class = SQUADNegRationaleDataLoader
    elif args.dataset_name in ('beer', 'hotel'):
        dataloader_class = SentimentDataLoader
    else:
        raise ValueError('Dataloader not implemented.')
    return dataloader_class


def save_args(args):
    with open(args.args_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f'Arg file saved at: {args.args_path}')


def save_ckpt(args, model, optimizer, latest=False):
    if not latest:
        args.best_ckpt_path = args.file_path.format(
            step=args.global_step,
            best_score=args.best_score * 100
        )
        checkpoint = {'ckpt_path': args.best_ckpt_path}
    else:
        checkpoint = {'ckpt_path': os.path.join(args.ckpt_dir, 'latest.ckpt')}

    checkpoint['args'] = vars(args)

    states = model.state_dict() if not args.dataparallel else model.module.state_dict()
    checkpoint['states'] = states
    checkpoint['optimizer_states'] = optimizer.state_dict()

    if not latest:
        for rm_path in glob.glob(os.path.join(args.ckpt_dir, '*.pt')):
            os.remove(rm_path)

    torch.save(checkpoint, checkpoint['ckpt_path'])
    print(f"Model saved at: {checkpoint['ckpt_path']}")


def load_ckpt(load_path):
    checkpoint = torch.load(load_path, map_location='cpu')
    args = Namespace(**checkpoint['args'])
    args = update_args(args)
    states = checkpoint['states']
    model_class = get_model_class(args)
    model = model_class(args)
    model.load_state_dict(states)
    model.eval()
    print('Model loaded from:', load_path)

    if 'optimizer_states' in checkpoint:
        optimizer_class = get_optimizer_class(args)
        optimizer = optimizer_class(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint['optimizer_states'])
        return model, args, optimizer
    else:
        return model, args, None


from torch.nn.parallel import DataParallel
import torch
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply

def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


"""Code taken from https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/utils/data_parallel.py"""
class BalancedDataParallel(DataParallel):
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)