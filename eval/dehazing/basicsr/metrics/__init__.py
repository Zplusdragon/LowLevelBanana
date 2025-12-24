from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .base import calculate_l1_loss
from .niqe import calculate_niqe
from .psnr_ssim import (calculate_psnr, calculate_ssim,
                        calculate_psnr_kornia, calculate_ssim_kornia)
from .CIEDE import calculate_CIEDE
from .nr_iqa import calculate_brisque, calculate_nima


__all__ = [
    'calculate_psnr', 'calculate_ssim', 'calculate_niqe',
    'calculate_psnr_kornia', 'calculate_ssim_kornia',
    'calculate_l1_loss', 'calculate_CIEDE',
    'calculate_brisque', 'calculate_nima'
]


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
