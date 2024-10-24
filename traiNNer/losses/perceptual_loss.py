import torch
from torch import Tensor, nn

from traiNNer.archs.vgg_arch import VGGFeatureExtractor
from traiNNer.losses.basic_loss import charbonnier_loss
from traiNNer.utils.registry import LOSS_REGISTRY

VGG_PATCH_SIZE = 256


class ProjectedDistributionLoss(nn.Module):
    def __init__(self, num_projections: int = 32) -> None:
        super().__init__()
        self.num_projections = num_projections
        self.criterion = nn.L1Loss()

    """Projected Distribution Loss (https://arxiv.org/abs/2012.09289)
    x.shape = B,M,N,...
    """

    def rand_projections(
        self,
        dim: int,
        device: torch.device,
        num_projections: int,
    ) -> Tensor:
        projections = torch.randn((dim, num_projections), device=device)
        projections = projections / torch.sqrt(
            torch.sum(projections**2, dim=0, keepdim=True)
        )  # columns are unit length normalized
        return projections

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], x.shape[1], -1)  # B,N,M
        y = y.reshape(y.shape[0], y.shape[1], -1)
        w = self.rand_projections(
            x.shape[-1], device=x.device, num_projections=self.num_projections
        )
        e_x = torch.matmul(x, w)
        e_y = torch.matmul(y, w)
        loss = torch.tensor(0.0, device=x.device)
        for ii in range(e_x.shape[2]):
            loss += self.criterion(
                torch.sort(e_x[:, :, ii], dim=1)[0], torch.sort(e_y[:, :, ii], dim=1)[0]
            )

        return loss * 2 / self.num_projections


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(
        self,
        layer_weights: dict[str, float],
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        range_norm: bool = False,
        normalize_layer_weights: bool = False,
        resize_input: bool = True,
        use_replicate_padding: bool = True,
        use_l2_pooling: bool = True,
        perceptual_weight: float = 1.0,
        style_weight: float = 0.0,
        criterion: str = "l1",
    ) -> None:
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight

        if normalize_layer_weights:
            layer_weights_sum = sum(layer_weights.values())
            for k, v in layer_weights.items():
                layer_weights[k] = v / layer_weights_sum

        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
            use_l2_pooling=use_l2_pooling,
            resize_input=resize_input,
            use_replicate_padding=use_replicate_padding,
        )

        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == "charbonnier":
            self.criterion = charbonnier_loss
        elif self.criterion_type == "pdl":
            self.criterion = ProjectedDistributionLoss()
        elif self.criterion_type == "fro":
            self.criterion = None
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, gt: Tensor) -> tuple[Tensor | None, Tensor | None]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = torch.tensor(0.0, device=x.device)
            for k in x_features.keys():
                if self.criterion is None:
                    percep_loss += (
                        torch.norm(x_features[k] - gt_features[k], p="fro")
                        * self.layer_weights[k]
                    )
                else:
                    percep_loss += (
                        self.criterion(x_features[k], gt_features[k])
                        * self.layer_weights[k]
                    )
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = torch.tensor(0.0, device=x.device)
            for k in x_features.keys():
                if self.criterion is None:
                    style_loss += (
                        torch.norm(
                            self._gram_mat(x_features[k])
                            - self._gram_mat(gt_features[k]),
                            p="fro",
                        )
                        * self.layer_weights[k]
                    )
                else:
                    style_loss += (
                        self.criterion(
                            self._gram_mat(x_features[k]),
                            self._gram_mat(gt_features[k]),
                        )
                        * self.layer_weights[k]
                    )
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x: Tensor) -> Tensor:
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
