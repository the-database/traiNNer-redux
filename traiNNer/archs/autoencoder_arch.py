from spandrel.architectures.__arch_helpers.block import RRDB
from spandrel.architectures.ESRGAN import ESRGAN
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class AutoEncoder(nn.Module):
    def __init__(self, freeze: bool, scale: int = 4, nf: int = 64) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # fromRGB
            nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(nf, nf // scale**2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # downscale
            nn.PixelUnshuffle(scale),
            RRDB(nf=nf),
            RRDB(nf=nf),
            # toRGB
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(nf, 3, kernel_size=3, stride=1, padding=1),
        )

        self.decoder = ESRGAN(scale=scale, num_filters=nf)

        if freeze:
            self.freeze()

    def forward(self, x: Tensor) -> Tensor:
        # TODO pad to divisible by scale? or unnecessary?
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
