import pyiqa

from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_brisque(
        img, # lq
        img2=None, # gt
):
    device = img.device

    iqa_metric = pyiqa.create_metric('brisque', device=device)

    score = iqa_metric(img) * -1.0

    return score.item()


@METRIC_REGISTRY.register()
def calculate_nima(
        img, # lq
        img2=None, # gt
):
    device = img.device

    iqa_metric = pyiqa.create_metric('nima', device=device)

    score = iqa_metric(img)

    return score.item()
