"""FlexAttention tile sizes for the window attention of HAT and DAT.

FlexAttention's default configs leave time on the table for these shapes (64 to 1024
tokens per window, head dim 32). Tuned on an RTX 5090 (compute capability 12.0): forward
-6 to -12%, backward -3 to -21% across HAT's 16 and 32 windows and its OCA, with bit-equal
results. They are used only on that architecture, as other GPUs weren't measured.
"""

from functools import cache

import torch

_SM120 = {
    "fwd_BLOCK_M": 128,
    "fwd_BLOCK_N": 64,
    "fwd_num_warps": 4,
    "fwd_num_stages": 2,
    "bwd_BLOCK_M1": 32,
    "bwd_BLOCK_N1": 64,
    "bwd_BLOCK_M2": 64,
    "bwd_BLOCK_N2": 32,
    "bwd_num_warps": 4,
    "bwd_num_stages": 2,
}


@cache
def flex_kernel_options(device: torch.device) -> dict[str, int] | None:
    """kernel_options for flex_attention on ``device``, or None for the defaults."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 12:
        return _SM120
    return None
