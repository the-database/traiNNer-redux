import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from traiNNer.archs.autoencoder_arch import AutoEncoder
from traiNNer.losses.ms_ssim_l1_loss import MSSSIML1Loss
from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class AESOPLoss(nn.Module):
    def __init__(
        self, loss_weight: float, scale: int, pretrain_network_ae: str
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.ae = AutoEncoder(freeze=True, scale=scale)
        self.ae.load_state_dict(
            load_file(pretrain_network_ae)
        )  # TODO wrapper function to support pth/safetensors
        self.criterion = MSSSIML1Loss(1.0)  # TODO support other loss?

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        with torch.inference_mode():
            ae_sr = self.ae(sr)
            ae_hr = self.ae(hr)
        return self.loss_weight * self.criterion(ae_sr, ae_hr)
