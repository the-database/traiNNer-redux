"""Per-device caches for small constant tensors used in hot paths.

Building a constant on the CPU and moving it to the GPU on every call (``torch.tensor(...).to(x)``)
is a blocking host-to-device copy, so the CPU waits for the GPU to drain its queue each time.
These helpers build each constant once per (args, device, dtype) and reuse it.
"""

from collections.abc import Callable, Hashable
from functools import wraps
from typing import cast

import torch
from torch import Tensor

_MAX_ENTRIES = 64


def cached_constant[**P](
    fn: Callable[P, Tensor],
) -> Callable[..., Tensor]:
    """Decorate a function that builds a constant CPU tensor from hashable args.

    The wrapped function takes the same positional args plus a keyword-only ``like``
    tensor (or ``device``/``dtype``) and returns the constant on that device and
    dtype, building it only on the first call for each key. Callers must not modify
    the returned tensor in place.
    """
    cache: dict[Hashable, Tensor] = {}

    @wraps(fn)
    def wrapper(
        *args: Hashable,
        like: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if like is not None:
            device = like.device if device is None else device
            dtype = like.dtype if dtype is None else dtype
        key = (args, device, dtype)
        t = cache.get(key)
        if t is None:
            if len(cache) >= _MAX_ENTRIES:
                cache.pop(next(iter(cache)))
            # A tensor made in inference mode can't be saved for backward later, so
            # build it as a normal tensor even when first requested there.
            with torch.inference_mode(False):
                t = fn(*args).to(device=device, dtype=dtype)  # pyright: ignore[reportCallIssue]
            cache[key] = t
        return t

    # Under torch.compile this runs eagerly (a graph break): traced, the CPU-side
    # construction would need a C++ compiler for Inductor's CPU kernels.
    return cast(Callable[..., Tensor], torch.compiler.disable(wrapper))
