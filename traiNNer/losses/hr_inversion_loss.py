# https://github.com/PeterouZh/HRInversion

from collections.abc import Callable, Iterable
from typing import Any, cast

import timm
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms as tv_trans
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import ClassifierHead
from timm.models import build_model_with_cfg, register_model
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]) -> None:
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_: Any, __: Any, output: Tensor) -> None:
            self._features[layer_id] = output

        return fn

    def forward(self, *args: Tensor, **kwargs) -> dict[str, Tensor]:
        self._features.clear()
        _ = self.model(*args, **kwargs)
        return self._features


def _cfg(url: str = "", **kwargs) -> dict[str, Any]:
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": (1, 1),
        "crop_pct": 0.875,
        "interpolation": "bilinear",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "features.0",
        "classifier": "head.fc",
        **kwargs,
    }


default_cfgs = {
    "vgg16": _cfg(url="https://download.pytorch.org/models/vgg16-397923af.pth"),
    "vgg19": _cfg(url="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"),
}


cfgs: dict[str, list[str | int]] = {
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class ConvMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        mlp_ratio: float,
        drop_rate: float,
        act_layer: type[nn.Module],
        conv_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        self.input_kernel_size = kernel_size
        mid_features = int(out_features * mlp_ratio)
        self.fc1 = conv_layer(in_features, mid_features, kernel_size, bias=True)
        self.act1 = act_layer(False)
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = conv_layer(mid_features, out_features, 1, bias=True)
        self.act2 = act_layer(False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-2] < self.input_kernel_size or x.shape[-1] < self.input_kernel_size:
            # keep the input size >= 7x7
            output_size = (
                max(self.input_kernel_size, x.shape[-2]),
                max(self.input_kernel_size, x.shape[-1]),
            )
            x = F.adaptive_avg_pool2d(x, output_size)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class VGG(nn.Module):
    def __init__(
        self,
        cfg: list[Any],
        num_classes: int = 1000,
        in_chans: int = 3,
        output_stride: int = 32,
        mlp_ratio: float = 1.0,
        act_layer: type[nn.Module] = nn.ReLU,
        conv_layer: type[nn.Module] = nn.Conv2d,
        norm_layer: nn.Module | None = None,
        global_pool: str = "avg",
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        assert output_stride == 32
        self.num_classes = num_classes
        self.num_features = 4096
        self.drop_rate = drop_rate
        self.feature_info = []
        prev_chs = in_chans
        net_stride = 1
        pool_layer = nn.MaxPool2d
        layers: list[nn.Module] = []
        for v in cfg:
            last_idx = len(layers) - 1
            if v == "M":
                self.feature_info.append(
                    {
                        "num_chs": prev_chs,
                        "reduction": net_stride,
                        "module": f"features.{last_idx}",
                    }
                )
                layers += [pool_layer(kernel_size=2, stride=2)]
                net_stride *= 2
            else:
                v = cast(int, v)
                conv2d = conv_layer(prev_chs, v, kernel_size=3, padding=1)
                if norm_layer is not None:
                    layers += [conv2d, norm_layer(v), act_layer(inplace=False)]
                else:
                    layers += [conv2d, act_layer(inplace=False)]
                prev_chs = v
        self.features = nn.Sequential(*layers)
        self.feature_info.append(
            {
                "num_chs": prev_chs,
                "reduction": net_stride,
                "module": f"features.{len(layers) - 1}",
            }
        )
        self.pre_logits = ConvMlp(
            prev_chs,
            self.num_features,
            7,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            act_layer=act_layer,
            conv_layer=conv_layer,
        )
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate
        )

        self._initialize_weights()

    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: str = "avg") -> None:
        self.num_classes = num_classes
        self.head = ClassifierHead(
            self.num_features,
            self.num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _filter_fn(state_dict: dict[str, Any]) -> dict[str, Any]:
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        k_r = k
        k_r = k_r.replace("classifier.0", "pre_logits.fc1")
        k_r = k_r.replace("classifier.3", "pre_logits.fc2")
        k_r = k_r.replace("classifier.6", "head.fc")
        if "classifier.0.weight" in k:
            v = v.reshape(-1, 512, 7, 7)  # noqa: PLW2901
        if "classifier.3.weight" in k:
            v = v.reshape(-1, 4096, 1, 1)  # noqa: PLW2901
        out_dict[k_r] = v
    return out_dict


def _create_vgg(variant: str, pretrained: bool, **kwargs: Any) -> nn.Module:
    cfg = variant.split("_")[0]
    # NOTE: VGG is one of the only models with stride==1 features, so indices are offset from other models
    out_indices = kwargs.get("out_indices", (0, 1, 2, 3, 4, 5))
    kwargs["pretrained_cfg"] = default_cfgs[variant]
    model = build_model_with_cfg(
        VGG,
        variant,
        pretrained,
        model_cfg=cfgs[cfg],
        feature_cfg={"flatten_sequential": True, "out_indices": out_indices},
        pretrained_filter_fn=_filter_fn,
        **kwargs,
    )
    return model


@register_model
def vgg16_conv(pretrained: bool = False, **kwargs: Any) -> nn.Module:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(**kwargs)
    return _create_vgg("vgg16", pretrained=pretrained, **model_args)


@register_model
def vgg19_conv(pretrained: bool = False, **kwargs: Any) -> nn.Module:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(**kwargs)
    return _create_vgg("vgg19", pretrained=pretrained, **model_args)


class VGG16ConvLoss(torch.nn.Module):
    def __init__(
        self,
        downsample_size: int = -1,
        fea_dict: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        if fea_dict is None:
            fea_dict = {
                "features_2": 0.0002,
                "features_7": 0.0001,
                "features_14": 0.0001,
                "features_21": 0.0002,
                "features_28": 0.0005,
            }

        self.downsample_size = downsample_size
        self.fea_dict = fea_dict

        self.mean = IMAGENET_DEFAULT_MEAN
        self.std = IMAGENET_DEFAULT_STD
        self.transform = tv_trans.Normalize(mean=self.mean, std=self.std)

        layers = list(fea_dict.keys())
        net = timm.create_model("vgg16_conv", pretrained=True, features_only=True)
        self.net = FeatureExtractor(net, layers=layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [-1 , 1]
        :param downsample_size:
        :param fea_dict:
        :param kwargs:
        :return:
        """

        self.net.eval()

        x = (x + 1) / 2.0
        x = self.transform(x)
        if self.downsample_size > 0:
            downsample_size = (self.downsample_size, self.downsample_size)
            x = F.interpolate(x, size=downsample_size, mode="area")

        feas_dict = self.net(x)
        feas = []
        for k, v in feas_dict.items():
            fea = v
            # b, c, h, w = fea.shape
            fea = fea.flatten(start_dim=1)
            fea = fea * self.fea_dict[k]
            feas.append(fea)
        feas = torch.cat(feas, dim=1)
        return feas


@LOSS_REGISTRY.register()
class HRInversionLoss(nn.Module):
    def __init__(
        self,
        # criterion: str = "l2",  # TODO
        loss_weight: float = 1.0,
        downsample_size: int = -1,
        fea_dict: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.vgg16_conv_loss = (
            VGG16ConvLoss(downsample_size=downsample_size, fea_dict=fea_dict)
            .to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            .requires_grad_(False)
        )
        self.loss_weight = loss_weight

        self.criterion = torch.nn.MSELoss(reduction="sum")

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor, gt: Tensor) -> Tensor:
        x_features = self.vgg16_conv_loss(x)
        gt_features = self.vgg16_conv_loss(gt)

        return self.criterion(x_features, gt_features) / x.shape[0] * self.loss_weight
