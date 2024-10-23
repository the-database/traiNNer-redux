import torch

# from geomloss import SamplesLoss
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torchvision.models import (
    EfficientNet_B7_Weights,
    Inception_V3_Weights,
    ResNet101_Weights,
    VGG19_Weights,
    efficientnet_b7,
    inception_v3,
    resnet101,
    vgg19,
)

from traiNNer.utils.registry import LOSS_REGISTRY


class VGG(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        vgg_pretrained_features = vgg19(weights=VGG19_Weights.DEFAULT).features
        assert isinstance(vgg_pretrained_features, nn.Sequential)

        # print(vgg_pretrained_features)
        self.stage1 = nn.Sequential()
        self.stage2 = nn.Sequential()
        self.stage3 = nn.Sequential()
        self.stage4 = nn.Sequential()
        self.stage5 = nn.Sequential()

        # vgg19
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        self.chns = [64, 128, 256, 512, 512]

    def get_features(self, x: Tensor) -> list[Tensor]:
        # normalize the data
        h = (x - self.mean) / self.std

        h = self.stage1(h)
        h_relu1_2 = h

        h = self.stage2(h)
        h_relu2_2 = h

        h = self.stage3(h)
        h_relu3_3 = h

        h = self.stage4(h)
        h_relu4_3 = h

        h = self.stage5(h)
        h_relu5_3 = h

        # get the features of each layer
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

        return outs

    def forward(self, x: Tensor) -> list[Tensor]:
        feats_x = self.get_features(x)
        return feats_x


class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        model = resnet101(weights=ResNet101_Weights.DEFAULT)
        model.eval()
        # print(model)

        self.stage1 = nn.Sequential(model.conv1, model.bn1, model.relu)
        self.stage2 = nn.Sequential(
            model.maxpool,
            model.layer1,
        )
        self.stage3 = nn.Sequential(
            model.layer2,
        )
        self.stage4 = nn.Sequential(
            model.layer3,
        )
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        self.chns = [64, 256, 512, 1024]

    def get_features(self, x: Tensor) -> list[Tensor]:
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]
        return outs

    def forward(self, x: Tensor) -> list[Tensor]:
        feats_x = self.get_features(x)
        return feats_x


class Inception(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False)

        # print(inception)
        self.stage1 = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
        )
        self.stage2 = nn.Sequential(
            inception.maxpool1,
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
        )
        self.stage3 = nn.Sequential(
            inception.maxpool2,
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
        )
        self.stage4 = nn.Sequential(
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        )
        for param in self.parameters():
            param.requires_grad = False

        self.chns = [64, 192, 288, 768]

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

    def get_features(self, x: Tensor) -> list[Tensor]:
        h = (x - self.mean) / self.std
        # h = (x-0.5)*2
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]
        return outs

    def forward(self, x: Tensor) -> list[Tensor]:
        feats_x = self.get_features(x)
        return feats_x


class EffNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = efficientnet_b7(weights=EfficientNet_B7_Weights).features  # [:6]
        model.eval()
        # print(model)
        self.stage1 = model[0:2]
        self.stage2 = model[2]
        self.stage3 = model[3]
        self.stage4 = model[4]
        self.stage5 = model[5]

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        for param in self.parameters():
            param.requires_grad = False
        self.chns = [32, 48, 80, 160, 224]

    def get_features(self, x: Tensor) -> list[Tensor]:
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        return outs

    def forward(self, x: Tensor) -> list[Tensor]:
        feats_x = self.get_features(x)
        return feats_x


@LOSS_REGISTRY.register()
class FDLoss(nn.Module):
    def __init__(
        self,
        patch_size: int = 5,
        stride: int = 1,
        num_proj: int = 256,
        model: str = "VGG",
        phase_weight: float = 1.0,
        loss_weight: float = 1,
    ) -> None:
        """
        patch_size, stride, num_proj: SWD slice parameters
        model: feature extractor, support VGG, ResNet, Inception, EffNet
        phase_weight: weight for phase branch
        """

        super().__init__()
        if model == "ResNet":
            self.model = ResNet()
        elif model == "EffNet":
            self.model = EffNet()
        elif model == "Inception":
            self.model = Inception()
        elif model == "VGG":
            self.model = VGG()
        else:
            raise ValueError(
                "Invalid model type! Valid models: VGG, Inception, EffNet, ResNet"
            )

        self.phase_weight = phase_weight
        self.stride = stride
        for i in range(len(self.model.chns)):
            rand = torch.randn(num_proj, self.model.chns[i], patch_size, patch_size)
            rand = rand / rand.view(rand.shape[0], -1).norm(dim=1).unsqueeze(
                1
            ).unsqueeze(2).unsqueeze(3)
            self.register_buffer(f"rand_{i}", rand)

        # self.geomloss = SamplesLoss()
        # print all the parameters

    def forward_once(self, x: Tensor, y: Tensor, idx: int = -1) -> Tensor:
        """
        x, y: input image tensors with the shape of (N, C, H, W)
        """
        rand = self.__getattr__(f"rand_{idx}")
        projx = F.conv2d(x, rand, stride=self.stride)
        projx = projx.reshape(projx.shape[0], projx.shape[1], -1)
        projy = F.conv2d(y, rand, stride=self.stride)
        projy = projy.reshape(projy.shape[0], projy.shape[1], -1)

        # sort the convolved input
        projx, _ = torch.sort(projx, dim=-1)
        projy, _ = torch.sort(projy, dim=-1)

        # compute the mean of the sorted convolved input
        s = torch.abs(projx - projy).mean([1, 2])
        return s

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: list[Tensor], y: list[Tensor]) -> Tensor:
        x = self.model(x)
        y = self.model(y)
        score = []  # torch.tensor(0.0, device=x[0].device)
        for i in range(len(x)):
            # Transform to Fourier Space
            fft_x = torch.fft.fftn(x[i], dim=(-2, -1))
            fft_y = torch.fft.fftn(y[i], dim=(-2, -1))

            # get the magnitude and phase of the extracted features
            x_mag = torch.abs(fft_x)
            x_phase = torch.angle(fft_x)
            y_mag = torch.abs(fft_y)
            y_phase = torch.angle(fft_y)

            x_mag = x_mag.reshape(x_mag.shape[0], x_mag.shape[1], -1)  # B,N,M
            y_mag = y_mag.reshape(y_mag.shape[0], y_mag.shape[1], -1)
            x_phase = x_phase.reshape(x_phase.shape[0], x_phase.shape[1], -1)  # B,N,M
            y_phase = y_phase.reshape(y_phase.shape[0], y_phase.shape[1], -1)

            s_amplitude = self.forward_once(x_mag, y_mag)
            s_phase = self.forward_once(x_phase, y_phase)

            # score += s_amplitude + s_phase * self.phase_weight
            score.append(s_amplitude + s_phase * self.phase_weight)

        return (
            sum(score).mean()
        )  # the bigger the score, the bigger the difference between the two images
