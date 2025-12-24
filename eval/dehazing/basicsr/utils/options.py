import os
import argparse
import yaml
from collections import OrderedDict
from os import path as osp
import datetime
import random

import torch
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
from basicsr.utils.misc import get_time_str, set_random_seed, scandir


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


# def parse(opt_path, is_train=True):
#     """Parse option file.

#     Args:
#         opt_path (str): Option file path.
#         is_train (str): Indicate whether in training or not. Default: True.

#     Returns:
#         (dict): Options.
#     """
#     with open(opt_path, mode='r') as f:
#         Loader, _ = ordered_yaml()
#         opt = yaml.load(f, Loader=Loader)

#     opt['is_train'] = is_train

#     if opt['num_gpu'] == 'auto':
#         opt['num_gpu'] = torch.cuda.device_count()

#     # datasets
#     for phase, dataset in opt['datasets'].items():
#         # for several datasets, e.g., test_1, test_2
#         phase = phase.split('_')[0]
#         dataset['phase'] = phase
#         if 'scale' in opt:
#             dataset['scale'] = opt['scale']
#         if dataset.get('dataroot_gt') is not None:
#             dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
#         if dataset.get('dataroot_lq') is not None:
#             dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

#     # paths
#     for key, val in opt['path'].items():
#         if (val is not None) and ('resume_state' in key
#                                   or 'pretrain_network' in key):
#             opt['path'][key] = osp.expanduser(val)
#     opt['path']['root'] = osp.abspath(
#         osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
#     if is_train:
#         experiments_root = osp.join(opt['path']['root'], 'experiments',
#                                     opt['name'])
#         opt['path']['experiments_root'] = experiments_root
#         opt['path']['models'] = osp.join(experiments_root, 'models')
#         opt['path']['training_states'] = osp.join(experiments_root,
#                                                   'training_states')
#         opt['path']['log'] = experiments_root
#         opt['path']['visualization'] = osp.join(experiments_root,
#                                                 'visualization')

#         # change some options for debug mode
#         if 'debug' in opt['name']:
#             if 'val' in opt:
#                 opt['val']['val_freq'] = 8
#             opt['logger']['print_freq'] = 1
#             opt['logger']['save_checkpoint_freq'] = 8
#     else:  # test
#         results_root = osp.join(opt['path']['root'], 'results', opt['name'])
#         opt['path']['results_root'] = results_root
#         opt['path']['log'] = results_root
#         opt['path']['visualization'] = osp.join(results_root, 'visualization')

#     return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':'
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + '\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])
    

def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    # parse yml to dict
    opt = yaml_load(args.opt)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, timeout=datetime.timedelta(seconds=86400), **opt['dist_params'])
        else:
            init_dist(args.launcher, timeout=datetime.timedelta(seconds=86400))
    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = is_train

    # debug setting
    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if is_train:
        experiments_root = opt['path'].get('experiments_root')
        if experiments_root is None:
            experiments_root = osp.join(root_path, 'experiments')
            experiments_root = osp.join(experiments_root, opt['name'])  

        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['tb_loger'] = osp.join(experiments_root, 'tb_logger')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 10
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 10

            if opt['train'].get('net_c2h_init_iters', 0) > 8:
                opt['train']['net_c2h_init_iters'] = 8
            
            if opt['train'].get('warmup_refine_iter', 0) > 2:
                opt['train']['warmup_refine_iter'] = 2

            if opt['datasets']['train'].get('progressive_training', False):
                progress_train_iters = opt['datasets']['train']['iters']
                progress_train_iters = [2 for _ in progress_train_iters]
                opt['val']['val_freq'] = sum(progress_train_iters)

                progress_train_iters[-1] += 10

                opt['datasets']['train']['iters'] = progress_train_iters
            

    else:  # test
        experiments_root = opt['path'].get('experiments_root')
        if experiments_root is None:
            experiments_root = osp.join(root_path, 'experiments')
            experiments_root = osp.join(experiments_root, opt['name']) 
            opt['path']['experiments_root'] = experiments_root
        
        if opt['path'].get('models') is None:
            opt['path']['models'] = osp.join(experiments_root, 'models')

        results_root = opt['path'].get('results_root')
        if results_root is None:
            results_root = osp.join(experiments_root, 'results_'+get_time_str())
            opt['path']['results_root'] = results_root

        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

        if opt['path'].get('pretrain_network_g') is None:
            if osp.isdir(opt['path']['models']):
                ckps = list(
                    scandir(opt['path']['models'], suffix='pth', recursive=False, full_path=False)
                )
                _ckps = []
                for _ckp in ckps:
                    if _ckp.startswith('net_g'):
                        _ckps.append(_ckp)
                ckps = _ckps
                
                _ckp_name = ckps[0].replace(ckps[0].split('_')[-1] ,"")
                if len(ckps) != 0:
                    ckps = [(v.split('_')[-1]).split('.pth')[0] for v in ckps]
                    if "latest" in ckps:
                        opt['path']['pretrain_network_g'] = osp.join(
                            opt['path']['models'], f"{_ckp_name}latest.pth"
                        )
                    else:
                        ckps = [float(v) for v in ckps]
                        opt['path']['pretrain_network_g'] = osp.join(
                            opt['path']['models'], f"{_ckp_name}{max(ckps):.0f}.pth"
                        )

        assert opt['path']['pretrain_network_g'] is not None, \
        "Network checkpoint path is required for test / inference."

    return opt, args


@master_only
def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)


@master_only
def save_opt_file(name, opt, experiments_root):
    # save the yml file to the experiment root
    import time
    t = str(time.strftime('%Y%m%d_%H%M%S', time.localtime()))
    filename = f'{name}_{t}.yml'
    filename = osp.join(experiments_root, filename)

    file_content = dict2str(opt)
    file_content = file_content.replace('None', '~')
    file_content = file_content.replace(opt['root_path'], '.')

    with open(filename, 'w') as f:
        f.write(file_content)
