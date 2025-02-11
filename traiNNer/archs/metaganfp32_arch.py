from torch import Tensor

from traiNNer.archs.metagan2_arch import MetaGan2
from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MetaGanFp32(MetaGan2):
    # @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
