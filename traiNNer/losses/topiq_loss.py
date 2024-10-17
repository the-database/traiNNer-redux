import torchvision.transforms.functional as tf
from torch import Tensor, nn

from traiNNer.archs.topiq_arch import CFANet
from traiNNer.losses.loss_util import weight_reduce_loss
from traiNNer.utils.registry import LOSS_REGISTRY

PATCH_SIZE_KADID = 384


@LOSS_REGISTRY.register()
class TOPIQLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0, resize_input: bool = True) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.loss = CFANet(
            semantic_model_name="resnet50",
            model_name="cfanet_fr_kadid_res50",
            use_ref=True,
        )
        self.loss.eval()
        for param in self.loss.parameters():
            param.requires_grad = False

        self.resize_input = resize_input

    # @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, gt: Tensor) -> Tensor:
        if self.resize_input:
            if x.shape[2] != PATCH_SIZE_KADID or x.shape[3] != PATCH_SIZE_KADID:
                assert x.shape == gt.shape
                x = tf.resize(
                    x,
                    [PATCH_SIZE_KADID],
                    interpolation=tf.InterpolationMode.BICUBIC,
                    antialias=True,
                )

                gt = tf.resize(
                    gt,
                    [PATCH_SIZE_KADID],
                    interpolation=tf.InterpolationMode.BICUBIC,
                    antialias=True,
                )

        return (1 - weight_reduce_loss(self.loss(x, gt))) * self.loss_weight
