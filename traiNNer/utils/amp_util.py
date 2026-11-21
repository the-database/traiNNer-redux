"""GradScaler helpers that keep the training step free of GPU-to-CPU syncs."""

import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

# growth_interval large enough that a scaler pinned at 1 never grows
_NEVER = 2**31 - 1


def make_grad_scaler(loss_scaling: bool, device: str = "cuda") -> GradScaler:
    """GradScaler for one optimizer.

    With ``loss_scaling`` (fp16 AMP) this is a normal dynamic loss scaler. Otherwise
    (bf16 AMP or fp32) the scale stays at 1, and the scaler is only used for its
    inf/NaN check: ``scaler.step`` skips the update when a gradient is non-finite.
    For fused optimizers that check runs on the GPU without syncing.
    """
    if loss_scaling:
        return GradScaler(device=device)
    return GradScaler(device=device, init_scale=1.0, growth_interval=_NEVER)


def update_grad_scaler(scaler: GradScaler, loss_scaling: bool) -> None:
    """``scaler.update()``, but a scaler without loss scaling is reset to 1 instead of
    backing off, since its scale only exists to drive the inf/NaN check."""
    if loss_scaling:
        scaler.update()
    else:
        scaler.update(1.0)


def grad_scaler_scale(scaler: GradScaler) -> Tensor:
    """Current scale as a GPU tensor, for logging. ``scaler.get_scale()`` would sync."""
    scale = scaler._get_scale_async()  # noqa: SLF001
    assert scale is not None
    return scale.detach().clone().to(torch.float32)


def guard_nonfinite(loss: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(loss_for_backward, is_finite)``.

    ``loss_for_backward`` equals ``loss`` while it is finite and is NaN otherwise, so
    a non-finite loss always gives non-finite gradients and the GradScaler skips the
    step, even when the bad term has no gradient path. Decided on the GPU, no sync.
    """
    finite = torch.isfinite(loss)
    return loss * torch.where(finite, 1.0, float("nan")), finite
