import torch
from torch import nn as nn
from torch.nn import functional as F

from ..archs.vgg_arch import VGGFeatureExtractor
from ..utils.registry import LOSS_REGISTRY


########################
# Contextual Loss
########################


def alt_layers_names(layers):
    new_layers = {}
    for k, v in layers.items():
        if "_" in k[:5]:
            new_k = k[:5].replace("_", "") + k[5:]
            new_layers[new_k] = v
    return new_layers


DIS_TYPES = ['cosine', 'l1', 'l2']


@LOSS_REGISTRY.register()
class ContextualLoss(nn.Module):
    """
    Contextual loss for unaligned images (https://arxiv.org/abs/1803.02077)

    https://github.com/roimehrez/contextualLoss
    https://github.com/S-aiueo32/ContextualLoss_pytorch
    https://github.com/z-bingo/Contextual-Loss-PyTorch

    layer_weights: is a dict, e.g., {'conv1_1': 1.0, 'conv3_2': 1.0}
    crop_quarter: boolean
    """

    def __init__(self,
                 loss_weight=1.0,
                 layer_weights={
                     "conv3_2": 1.0,
                     "conv4_2": 1.0
                 },
                 crop_quarter: bool = False,
                 max_1d_size: int = 100,
                 distance_type: str = 'cosine',
                 b=1.0,
                 band_width=0.5,
                 use_vgg: bool = True,
                 net: str = 'vgg19',
                 calc_type: str = 'regular',
                 z_norm: bool = False):
        super(ContextualLoss, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        assert distance_type in DIS_TYPES, \
            f'select a distance type from {DIS_TYPES}.'

        if layer_weights:
            layer_weights = alt_layers_names(layer_weights)
            self.layer_weights = layer_weights
            listen_list = list(layer_weights.keys())
        else:
            listen_list = []
            self.layer_weights = {}

        self.loss_weight = loss_weight
        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size
        self.b = b
        self.band_width = band_width  # self.h = h, #sigma

        if use_vgg:
            self.vgg_model = VGGFeatureExtractor(
                layer_name_list=listen_list, vgg_type=net, use_input_norm=z_norm, range_norm=z_norm)

        if calc_type == 'bilateral':
            self.calculate_loss = self.bilateral_CX_Loss
        elif calc_type == 'symetric':
            self.calculate_loss = self.symetric_CX_Loss
        else:  # if calc_type == 'regular':
            self.calculate_loss = self.calculate_CX_Loss

    def forward(self, images, gt):
        device = images.device

        if hasattr(self, 'vgg_model'):
            assert images.shape[1] == 3 and gt.shape[1] == 3, \
                'VGG model takes 3 channel images.'

            loss = 0
            vgg_images = self.vgg_model(images)
            vgg_images = {k: v.clone().to(device) for k, v in vgg_images.items()}
            vgg_gt = self.vgg_model(gt)
            vgg_gt = {k: v.to(device) for k, v in vgg_gt.items()}

            for key in self.layer_weights.keys():
                if self.crop_quarter:
                    vgg_images[key] = self._crop_quarters(vgg_images[key])
                    vgg_gt[key] = self._crop_quarters(vgg_gt[key])

                N, C, H, W = vgg_images[key].size()
                if H * W > self.max_1d_size ** 2:
                    vgg_images[key] = self._random_pooling(vgg_images[key], output_1d_size=self.max_1d_size)
                    vgg_gt[key] = self._random_pooling(vgg_gt[key], output_1d_size=self.max_1d_size)

                loss_t = self.calculate_loss(vgg_images[key], vgg_gt[key])
                loss += loss_t * self.layer_weights[key]
                # del vgg_images[key], vgg_gt[key]
        # TODO: without VGG it runs, but results are not looking right
        else:
            if self.crop_quarter:
                images = self._crop_quarters(images)
                gt = self._crop_quarters(gt)

            N, C, H, W = images.size()
            if H * W > self.max_1d_size ** 2:
                images = self._random_pooling(images, output_1d_size=self.max_1d_size)
                gt = self._random_pooling(gt, output_1d_size=self.max_1d_size)

            loss = self.calculate_loss(images, gt)
        return loss

    @staticmethod
    def _random_sampling(tensor, n, indices):
        N, C, H, W = tensor.size()
        S = H * W
        tensor = tensor.view(N, C, S)
        device = tensor.device
        if indices is None:
            indices = torch.randperm(S)[:n].contiguous().type_as(tensor).long()
            indices = indices.clamp(indices.min(), tensor.shape[-1] - 1)  # max = indices.max()-1
            indices = indices.view(1, 1, -1).expand(N, C, -1)
        indices = indices.to(device)

        res = torch.gather(tensor, index=indices, dim=-1)
        return res, indices

    @staticmethod
    def _random_pooling(feats, output_1d_size=100):
        single_input = type(feats) is torch.Tensor

        if single_input:
            feats = [feats]

        N, C, H, W = feats[0].size()
        feats_sample, indices = ContextualLoss._random_sampling(feats[0], output_1d_size ** 2, None)
        res = [feats_sample]

        for i in range(1, len(feats)):
            feats_sample, _ = ContextualLoss._random_sampling(feats[i], -1, indices)
            res.append(feats_sample)

        res = [feats_sample.view(N, C, output_1d_size, output_1d_size) for feats_sample in res]

        if single_input:
            return res[0]
        return res

    @staticmethod
    def _crop_quarters(feature_tensor):
        N, fC, fH, fW = feature_tensor.size()
        quarters_list = []
        quarters_list.append(feature_tensor[..., 0:round(fH / 2), 0:round(fW / 2)])
        quarters_list.append(feature_tensor[..., 0:round(fH / 2), round(fW / 2):])
        quarters_list.append(feature_tensor[..., round(fH / 2):, 0:round(fW / 2)])
        quarters_list.append(feature_tensor[..., round(fH / 2):, round(fW / 2):])

        feature_tensor = torch.cat(quarters_list, dim=0)
        return feature_tensor

    @staticmethod
    def _create_using_L2(I_features, T_features):
        """
        Calculating the distance between each feature of I and T
        :param I_features:
        :param T_features:
        :return: raw_distance: [N, C, H, W, H*W], each element of which is the distance between I and T at each position
        """
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)

        square_I = torch.sum(Ivecs * Ivecs, dim=1, keepdim=False)
        square_T = torch.sum(Tvecs * Tvecs, dim=1, keepdim=False)
        # raw_distance
        raw_distance = []
        for i in range(N):
            Ivec, Tvec, s_I, s_T = Ivecs[i, ...], Tvecs[i, ...], square_I[i, ...], square_T[i, ...]
            # matrix multiplication
            AB = Ivec.permute(1, 0) @ Tvec
            dist = s_I.view(-1, 1) + s_T.view(1, -1) - 2 * AB
            raw_distance.append(dist.view(1, H, W, H * W))
        raw_distance = torch.cat(raw_distance, dim=0)
        raw_distance = torch.clamp(raw_distance, 0.0)
        return raw_distance

    @staticmethod
    def _create_using_L1(I_features, T_features):
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)

        raw_distance = []
        for i in range(N):
            Ivec, Tvec = Ivecs[i, ...], Tvecs[i, ...]
            dist = torch.sum(torch.abs(Ivec.view(C, -1, 1) - Tvec.view(C, 1, -1)), dim=0, keepdim=False)
            raw_distance.append(dist.view(1, H, W, H * W))
        raw_distance = torch.cat(raw_distance, dim=0)
        return raw_distance

    @staticmethod
    def _create_using_dotP(I_features, T_features):
        assert I_features.size() == T_features.size()
        # prepare feature before calculating cosine distance
        # mean shifting by channel-wise mean of `y`.
        mean_T = T_features.mean(dim=(0, 2, 3), keepdim=True)
        I_features = I_features - mean_T
        T_features = T_features - mean_T

        # L2 channelwise normalization
        I_features = F.normalize(I_features, p=2, dim=1)
        T_features = F.normalize(T_features, p=2, dim=1)

        N, C, H, W = I_features.size()
        cosine_dist = []
        # work seperatly for each example in dim 1
        for i in range(N):
            # channel-wise vectorization
            T_features_i = T_features[i].view(1, 1, C, H * W).permute(3, 2, 0,
                                                                      1).contiguous()  # 1CHW --> 11CP, with P=H*W
            I_features_i = I_features[i].unsqueeze(0)
            dist = F.conv2d(I_features_i, T_features_i).permute(0, 2, 3, 1).contiguous()
            # cosine_dist.append(dist) # back to 1CHW
            # TODO: temporary hack to workaround AMP bug:
            cosine_dist.append(dist.to(torch.float32))  # back to 1CHW
        cosine_dist = torch.cat(cosine_dist, dim=0)
        cosine_dist = (1 - cosine_dist) / 2
        cosine_dist = cosine_dist.clamp(min=0.0)

        return cosine_dist

    # compute_relative_distance
    @staticmethod
    def _calculate_relative_distance(raw_distance, epsilon=1e-5):
        """
        Normalizing the distances first as Eq. (2) in paper
        :param raw_distance:
        :param epsilon:
        :return:
        """
        div = torch.min(raw_distance, dim=-1, keepdim=True)[0]
        relative_dist = raw_distance / (div + epsilon)  # Eq 2
        return relative_dist

    def symetric_CX_Loss(self, I_features, T_features):
        loss = (self.calculate_CX_Loss(T_features, I_features) + self.calculate_CX_Loss(I_features, T_features)) / 2
        return loss * self.loss_weight  # score

    def bilateral_CX_Loss(self, I_features, T_features, weight_sp: float = 0.1):

        def compute_meshgrid(shape):
            N, C, H, W = shape
            rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
            cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

            feature_grid = torch.meshgrid(rows, cols)
            feature_grid = torch.stack(feature_grid).unsqueeze(0)
            feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

            return feature_grid

        # spatial loss
        grid = compute_meshgrid(I_features.shape).to(T_features.device)
        raw_distance = ContextualLoss._create_using_L2(grid, grid)  # calculate raw distance
        dist_tilde = ContextualLoss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - dist_tilde) / self.band_width)  # Eq(3)
        cx_sp = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)  # Eq(4)

        # feature loss
        # calculate raw distances
        if self.distanceType == 'l1':
            raw_distance = ContextualLoss._create_using_L1(I_features, T_features)
        elif self.distanceType == 'l2':
            raw_distance = ContextualLoss._create_using_L2(I_features, T_features)
        else:  # self.distanceType == 'cosine':
            raw_distance = ContextualLoss._create_using_dotP(I_features, T_features)
        dist_tilde = ContextualLoss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - dist_tilde) / self.band_width)  # Eq(3)
        cx_feat = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)  # Eq(4)

        # combined loss
        cx_combine = (1. - weight_sp) * cx_feat + weight_sp * cx_sp
        k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)
        cx = k_max_NC.mean(dim=1)
        cx_loss = torch.mean(-torch.log(cx + 1e-5))
        return cx_loss * self.loss_weight

    def calculate_CX_Loss(self, I_features, T_features):
        device = I_features.device
        T_features = T_features.to(device)

        if torch.sum(torch.isnan(I_features)) == torch.numel(I_features) or torch.sum(
                torch.isinf(I_features)) == torch.numel(I_features):
            print(I_features)
            raise ValueError('NaN or Inf in I_features')
        if torch.sum(torch.isnan(T_features)) == torch.numel(T_features) or torch.sum(
                torch.isinf(T_features)) == torch.numel(T_features):
            print(T_features)
            raise ValueError('NaN or Inf in T_features')

        # calculate raw distances
        if self.distanceType == 'l1':
            raw_distance = ContextualLoss._create_using_L1(I_features, T_features)
        elif self.distanceType == 'l2':
            raw_distance = ContextualLoss._create_using_L2(I_features, T_features)
        else:  # self.distanceType == 'cosine':
            raw_distance = ContextualLoss._create_using_dotP(I_features, T_features)
        if torch.sum(torch.isnan(raw_distance)) == torch.numel(raw_distance) or torch.sum(
                torch.isinf(raw_distance)) == torch.numel(raw_distance):
            print(raw_distance)
            raise ValueError('NaN or Inf in raw_distance')

        # normalizing the distances
        relative_distance = ContextualLoss._calculate_relative_distance(raw_distance)
        if torch.sum(torch.isnan(relative_distance)) == torch.numel(relative_distance) or torch.sum(
                torch.isinf(relative_distance)) == torch.numel(relative_distance):
            print(relative_distance)
            raise ValueError('NaN or Inf in relative_distance')
        del raw_distance

        # compute_sim()
        # where h>0 is a band-width parameter
        exp_distance = torch.exp((self.b - relative_distance) / self.band_width)  # Eq(3)
        if torch.sum(torch.isnan(exp_distance)) == torch.numel(exp_distance) or torch.sum(
                torch.isinf(exp_distance)) == torch.numel(exp_distance):
            print(exp_distance)
            raise ValueError('NaN or Inf in exp_distance')
        del relative_distance

        # Similarity
        contextual_sim = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)  # Eq(4)
        if torch.sum(torch.isnan(contextual_sim)) == torch.numel(contextual_sim) or torch.sum(
                torch.isinf(contextual_sim)) == torch.numel(contextual_sim):
            print(contextual_sim)
            raise ValueError('NaN or Inf in contextual_sim')
        del exp_distance

        # ContextualLoss()
        max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0]  # Eq(1)
        del contextual_sim
        CS = torch.mean(max_gt_sim, dim=1)
        CX_loss = torch.mean(-torch.log(CS))  # Eq(5)
        if torch.isnan(CX_loss):
            raise ValueError('NaN in computing CX_loss')

        return CX_loss * self.loss_weight
