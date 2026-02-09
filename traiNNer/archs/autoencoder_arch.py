from typing import Any

from spandrel.architectures.__arch_helpers.block import RRDB
from torch import Tensor, nn

from traiNNer.archs.arch_util import default_init_weights
from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class AutoEncoder(nn.Module):
    def __init__(
        self,
        decoder_opt: dict[str, Any],
        freeze_decoder: bool,
        freeze_encoder: bool,
        scale: int = 4,
        nf: int = 64,
    ) -> None:
        super().__init__()

        # encoder
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, nf // scale**2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(
                nf // scale**2, nf // scale**2, kernel_size=3, stride=1, padding=1
            ),
        )
        self.down = nn.PixelUnshuffle(scale)
        self.body = nn.Sequential(RRDB(nf=nf), RRDB(nf=nf))
        self.conv_last = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(nf, 3, kernel_size=3, stride=1, padding=1),
        )

        self.encoder = nn.Sequential(
            self.conv_first, self.down, self.body, self.conv_last
        )

        from traiNNer.models.sr_model import build_network

        self.decoder = build_network({**decoder_opt, "scale": scale})

        default_init_weights([self.conv_first, self.conv_last], 0.1)

        self.encoder_is_frozen = False
        self.decoder_is_frozen = False

        if freeze_encoder:
            self.freeze_encoder()

        if freeze_decoder:
            self.freeze_decoder()

    def forward(self, x: Tensor) -> Tensor:
        # TODO pad to divisible by scale? or unnecessary?
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder_is_frozen = True

    def freeze_decoder(self) -> None:
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()
        self.decoder_is_frozen = True

    def train(self, mode: bool = True) -> AutoEncoder:
        super().train(mode)
        if self.decoder_is_frozen:
            self.decoder.eval()
        return self
