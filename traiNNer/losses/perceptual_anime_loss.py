# https://github.com/Kiteretsu77/APISR/blob/main/loss/anime_perceptual_loss.py
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torchvision import models

from traiNNer.losses.basic_loss import charbonnier_loss
from traiNNer.utils.registry import LOSS_REGISTRY


class AdaptiveConcatPool2d(nn.Module):
    """
    Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`.
    Source: Fastai. This code was taken from the fastai library at url
    https://github.com/fastai/fastai/blob/master/fastai/layers.py#L176
    """

    def __init__(self, sz=None) -> None:
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    """
    Flatten `x` to a single dimension. Adapted from fastai's Flatten() layer,
    at https://github.com/fastai/fastai/blob/master/fastai/layers.py#L25
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


def bn_drop_lin(
    n_in: int,
    n_out: int,
    bn: bool = True,
    p: float = 0.0,
    actn: nn.Module | None = None,
) -> list[nn.Module]:
    """
    Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`.
    Adapted from Fastai at https://github.com/fastai/fastai/blob/master/fastai/layers.py#L44
    """
    layers: list[nn.Module] = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers


def create_head(top_n_tags: int, nf: int, ps: float = 0.5) -> nn.Sequential:
    nc = top_n_tags

    lin_ftrs = [nf, 512, nc]
    p1 = 0.25  # dropout for second last layer
    p2 = 0.5  # dropout for last layer

    # _actns = [nn.ReLU(inplace=True), None]
    pool = AdaptiveConcatPool2d()
    layers = [pool, Flatten()]

    layers += [
        *bn_drop_lin(lin_ftrs[0], lin_ftrs[1], True, p1, nn.ReLU(inplace=True)),
        *bn_drop_lin(lin_ftrs[1], lin_ftrs[2], True, p2),
    ]

    return nn.Sequential(*layers)


def _resnet(base_arch: Callable[..., models.ResNet], top_n: int, **kwargs) -> nn.Module:
    cut = -2
    s = base_arch(weights=None, **kwargs)
    body = nn.Sequential(*list(s.children())[:cut])

    model = body  # nn.Sequential(body, head)

    return model


def resnet50(
    pretrained: bool = True, progress: bool = True, top_n: int = 6000, **kwargs
) -> nn.Module:
    r"""
    Resnet50 model trained on the full Danbooru2018 dataset's top 6000 tags

    Args:
        pretrained (bool): kwargs, load pretrained weights into the model.
        top_n (int): kwargs, pick to load the model for predicting the top `n` tags,
            currently only supports top_n=6000.
    """
    model = _resnet(
        models.resnet50, top_n, **kwargs
    )  # Take Resnet without the head (we don't care about final FC layers)

    if pretrained:
        if top_n == 6000:
            state = torch.hub.load_state_dict_from_url(
                "https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet50-13306192.pth",
                progress=progress,
            )
            old_keys = list(state)
            for old_key in old_keys:
                if old_key[0] == "0":
                    new_key = old_key[2:]
                    state[new_key] = state[old_key]
                    del state[old_key]
                elif old_key[0] == "1":
                    del state[old_key]

            model.load_state_dict(state)
        else:
            raise ValueError(
                "Sorry, the resnet50 model only supports the top-6000 tags \
                at the moment"
            )

    return model


class ResNet50Extractor(nn.Module):
    """ResNet50 network for feature extraction."""

    def get_activation(self, name: str) -> Callable[..., None]:
        def hook(_model: nn.Module, _input: Tensor, output: Tensor) -> None:
            self.activation[name] = output

        return hook

    def __init__(
        self,
        model: nn.Module,
        layer_labels: Sequence[str],
        use_input_norm: bool = True,
        range_norm: bool = False,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.model = model
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        self.layer_labels = layer_labels
        self.activation = {}

        # Extract needed features
        for layer_label in layer_labels:
            elements = layer_label.split("_")
            if len(elements) == 1:
                # modified_net[layer_label] = getattr(model, elements[0])
                getattr(self.model, elements[0]).register_forward_hook(
                    self.get_activation(layer_label)
                )
            else:
                body_layer = self.model
                for element in elements[:-1]:
                    # Iterate until the last element
                    assert isinstance(int(element), int)
                    body_layer = body_layer[int(element)]  # pyright: ignore[reportIndexIssue]
                getattr(body_layer, elements[-1]).register_forward_hook(
                    self.get_activation(layer_label)
                )

        # Set as evaluation
        if not requires_grad:
            self.model.eval()
            for param in self.parameters():
                param.requires_grad = False

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            # the std is for image with range [0, 1]
            self.register_buffer(
                "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std  # pyright: ignore[reportOperatorIssue]

        # Execute model first
        self.model(x)  # Zomby input

        # Extract the layers we need
        store = {}
        for layer_label in self.layer_labels:
            store[layer_label] = self.activation[layer_label]

        return store


@LOSS_REGISTRY.register()
class PerceptualAnimeLoss(nn.Module):
    """Anime Perceptual loss

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(
        self,
        loss_weight: float,
        layer_weights: dict[str, float] | None = None,
        criterion: str = "l1",
    ) -> None:
        super().__init__()

        if layer_weights is None:
            layer_weights = {
                "0": 0.1,
                "4_2_conv3": 20,
                "5_3_conv3": 25,
                "6_5_conv3": 1,
                "7_2_conv3": 1,
            }

        model = resnet50()
        self.loss_weight = loss_weight
        self.layer_weights = layer_weights
        self.layer_labels = layer_weights.keys()
        self.resnet50 = ResNet50Extractor(model, list(self.layer_labels)).cuda()

        if criterion == "l1":
            self.criterion = torch.nn.L1Loss()
        elif criterion == "charbonnier":
            self.criterion1 = charbonnier_loss
        else:
            raise NotImplementedError(
                "We don't support such criterion loss in perceptual loss"
            )

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, gen: Tensor, gt: Tensor) -> dict[str, Tensor]:
        """Forward function.

        Args:
            gen (Tensor):   Input tensor with shape (n, c, h, w).
            gt (Tensor):    Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        gen_features = self.resnet50(gen)
        gt_features = self.resnet50(gt.detach())

        # calculate perceptual loss
        losses = {}
        for _idx, k in enumerate(gen_features.keys()):
            raw_comparison = self.criterion(gen_features[k], gt_features[k])
            layer_loss = raw_comparison * self.layer_weights[k]
            losses[k] = layer_loss * self.loss_weight

        return losses
