import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_l1_loss(
        img, # lq
        img2, # gt
        max_val=1.0,
        **args
):
    return F.l1_loss(img, img2, **args).item()