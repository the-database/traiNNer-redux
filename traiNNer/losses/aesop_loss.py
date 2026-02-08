from typing import Literal

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from traiNNer.archs.autoencoder_arch import AutoEncoder
from traiNNer.losses.basic_loss import L1Loss, charbonnier_loss
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class AESOPLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float,
        scale: int,
        pretrain_network_ae: str,
        criterion: Literal["l1", "charbonnier"] = "charbonnier",
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.ae = AutoEncoder(freeze_encoder=True, freeze_decoder=True, scale=scale)
        self.ae.load_state_dict(
            load_file(pretrain_network_ae)
        )  # TODO wrapper function to support pth/safetensors
        self.ae.eval()
        if criterion == "l1":
            self.criterion = L1Loss(1.0)
        elif criterion == "charbonnier":
            self.criterion = charbonnier_loss
        else:
            raise NotImplementedError(f"invalid criterion: {criterion}")

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        with torch.no_grad():
            ae_hr = self.ae(hr.detach())
        ae_sr = self.ae(sr)
        return self.criterion(ae_sr, ae_hr)
