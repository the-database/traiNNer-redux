# https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
# https://github.com/neosr-project/neosr/blob/master/neosr/archs/patchgan_arch.py
import torch
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MultiscalePatchGANDiscriminatorSN(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_sigmoid: bool = False,
        num_d: int = 3,
    ) -> None:
        super().__init__()
        self.num_d = num_d

        self.layers = nn.ModuleList(
            [
                PatchGANDiscriminatorSN(input_nc, ndf, n_layers, use_sigmoid).model
                for _ in range(num_d)
            ]
        )

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=(1, 1), count_include_pad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        num_d = self.num_d
        result = []
        input_downsampled = x
        for i, model in enumerate(self.layers):
            result.append(model(input_downsampled).mean())
            if i != (num_d - 1):
                input_downsampled = self.downsample(input_downsampled)
        return torch.stack(result).mean()


@ARCH_REGISTRY.register()
# Defines the PatchGAN discriminator with the specified arguments.
class PatchGANDiscriminatorSN(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        norm = spectral_norm

        kw = 4
        padw = (kw - 1) // 2
        sequence = [
            [
                norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for _n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    norm(
                        nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)
                    ),
                    nn.LeakyReLU(0.2, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        sequence += [[norm(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw))]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
