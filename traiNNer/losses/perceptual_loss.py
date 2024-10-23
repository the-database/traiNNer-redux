import torch
import torchvision
from torch import Tensor, nn

from traiNNer.archs.vgg_arch import VGGFeatureExtractor
from traiNNer.losses.basic_loss import charbonnier_loss
from traiNNer.utils.registry import LOSS_REGISTRY

VGG_PATCH_SIZE = 256


class PDLoss(nn.Module):
    def __init__(self, w_lambda: float = 0.01) -> None:
        super().__init__()
        self.w_lambda = w_lambda

    def w_distance(self, x_feat: Tensor, y_feat: Tensor) -> Tensor:
        # print(
        #     "w_distance 0",
        #     x_feat.shape,
        #     y_feat.shape,
        #     x_feat.min(),
        #     x_feat.max(),
        #     y_feat.min(),
        #     y_feat.max(),
        # )
        x_feat = x_feat / (torch.sum(x_feat, dim=(2, 3), keepdim=True) + 1e-14)
        y_feat = y_feat / (torch.sum(y_feat, dim=(2, 3), keepdim=True) + 1e-14)

        # print(
        #     "w_distance 1",
        #     x_feat.shape,
        #     y_feat.shape,
        #     x_feat.min(),
        #     x_feat.max(),
        #     y_feat.min(),
        #     y_feat.max(),
        # )

        x_feat = x_feat.view(x_feat.size()[0], x_feat.size()[1], -1).contiguous()
        y_feat = y_feat.view(y_feat.size()[0], y_feat.size()[1], -1).contiguous()

        # print(
        #     "w_distance 2",
        #     x_feat.shape,
        #     y_feat.shape,
        #     x_feat.min(),
        #     x_feat.max(),
        #     y_feat.min(),
        #     y_feat.max(),
        # )

        cdf_x_feat = torch.cumsum(x_feat, dim=-1)
        cdf_y_feat = torch.cumsum(y_feat, dim=-1)

        # print(
        #     "w_distance 3 (cdf feat)",
        #     cdf_x_feat.shape,
        #     cdf_y_feat.shape,
        #     cdf_x_feat.min(),
        #     cdf_x_feat.max(),
        #     cdf_y_feat.min(),
        #     cdf_y_feat.max(),
        # )

        cdf_distance = torch.sum(torch.abs(cdf_x_feat - cdf_y_feat), dim=-1)
        # print(
        #     "w_distance 4 (cdf dist)",
        #     cdf_distance.shape,
        #     cdf_distance.min(),
        #     cdf_distance.max(),
        # )
        cdf_loss = cdf_distance.mean()
        # print("?", cdf_loss)
        return cdf_loss

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        out = self.w_distance(x, y) * self.w_lambda
        # print("?", out)
        return out


@LOSS_REGISTRY.register()
class PDLossStandalone(nn.Module):
    def __init__(self, l1_lambda=1.5, w_lambda=0.01, loss_weight=1) -> None:
        super().__init__()
        # self.vgg = Vgg19Conv4().cuda()
        # self.layer_weights = {
        #     "conv1_2": 0.1,
        #     "conv2_2": 0.1,
        #     "conv3_4": 1,
        #     "conv4_4": 1,
        #     "conv5_4": 1,
        # }
        # self.vgg = VGGFeatureExtractor(layer_name_list=list(self.layer_weights.keys()))
        self.vgg = Vgg19Conv4().cuda()
        self.criterionL1 = nn.L1Loss()
        self.w_lambda = w_lambda
        self.l1_lambda = l1_lambda

        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def w_distance(self, xvgg, yvgg):
        # print(
        #     "w_distance 0",
        #     xvgg.shape,
        #     yvgg.shape,
        #     xvgg.min(),
        #     xvgg.max(),
        #     yvgg.min(),
        #     yvgg.max(),
        # )
        xvgg = xvgg / (torch.sum(xvgg, dim=(2, 3), keepdim=True) + 1e-14)
        yvgg = yvgg / (torch.sum(yvgg, dim=(2, 3), keepdim=True) + 1e-14)

        # print(
        #     "w_distance 1",
        #     xvgg.shape,
        #     yvgg.shape,
        #     xvgg.min(),
        #     xvgg.max(),
        #     yvgg.min(),
        #     yvgg.max(),
        # )

        xvgg = xvgg.view(xvgg.size()[0], xvgg.size()[1], -1).contiguous()
        yvgg = yvgg.view(yvgg.size()[0], yvgg.size()[1], -1).contiguous()

        # print(
        #     "w_distance 2",
        #     xvgg.shape,
        #     yvgg.shape,
        #     xvgg.min(),
        #     xvgg.max(),
        #     yvgg.min(),
        #     yvgg.max(),
        # )

        cdf_xvgg = torch.cumsum(xvgg, dim=-1)
        cdf_yvgg = torch.cumsum(yvgg, dim=-1)

        # print(
        #     "w_distance 3 (cdf feat)",
        #     cdf_xvgg.shape,
        #     cdf_yvgg.shape,
        #     cdf_xvgg.min(),
        #     cdf_xvgg.max(),
        #     cdf_yvgg.min(),
        #     cdf_yvgg.max(),
        # )

        cdf_distance = torch.sum(torch.abs(cdf_xvgg - cdf_yvgg), dim=-1)

        # print(
        #     "w_distance 4 (cdf mean)",
        #     cdf_distance.shape,
        #     cdf_distance.min(),
        #     cdf_distance.max(),
        # )

        cdf_loss = cdf_distance.mean()

        return cdf_loss

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x, y):
        print("standalone 0", x.shape, x.min(), x.max())
        # L1loss = self.criterionL1(x, y) * self.l1_lambda
        # L1loss = 0
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        print("standalone 1", x.shape, x.min(), x.max())
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        print("standalone 2", x_vgg.shape, x_vgg.min(), x_vgg.max())

        # percep_loss = torch.tensor(0.0, device=x.device)
        # for k in x_vgg.keys():
        #     percep_loss += self.w_distance(x_vgg[k], y_vgg[k]) * self.layer_weights[k]

        WdLoss = self.w_distance(x_vgg, y_vgg) * self.w_lambda
        print("?", WdLoss)
        return WdLoss


# ### Define Vgg19 for projected distribution loss
class Vgg19Conv4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        assert isinstance(vgg_pretrained_features, torch.nn.Sequential)
        self.slice1 = nn.Sequential()

        for x in range(23):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(4):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # self.slice1.add_module(str(4), L2pooling(channels=64, as_loss=True))
        # for x in range(5, 9):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # self.slice1.add_module(str(9), L2pooling(channels=128, as_loss=True))
        # for x in range(10, 18):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # self.slice1.add_module(str(16), L2pooling(channels=256, as_loss=True))
        # for x in range(19, 27):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # self.slice1.add_module(str(23), L2pooling(channels=512, as_loss=True))
        # for x in range(28, 36):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])

        # fixed pretrained vgg19 model for feature extraction
        for param in self.parameters():
            param.requires_grad = False

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x):
        out = self.slice1(x)
        return out


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
            # self.criterion = PDLoss()
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
                    # print("perceptual", x.shape, x.min(), x.max())
                    # print(
                    #     f"feat {k}",
                    #     x_features[k].shape,
                    #     x_features[k].min(),
                    #     x_features[k].max(),
                    #     gt_features[k].shape,
                    #     gt_features[k].min(),
                    #     gt_features[k].max(),
                    # )
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
