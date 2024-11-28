from copy import deepcopy
from typing import Any

import torch

from traiNNer.metrics.dists import calculate_dists
from traiNNer.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from traiNNer.utils.registry import METRIC_REGISTRY

__all__ = ["calculate_dists", "calculate_psnr", "calculate_ssim"]


def calculate_metric(
    data: dict[str, Any], opt: dict[str, Any], device: torch.device
) -> float:
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop("type")
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt, device=device)
    return metric
