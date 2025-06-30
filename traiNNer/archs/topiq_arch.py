"""TOP-IQ metric, proposed by

TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment.
Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin.
Transactions on Image Processing, 2024.

Paper link: https://arxiv.org/abs/2308.03060

"""

import copy
import os
from collections.abc import Callable
from urllib.parse import urlparse

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.functional as TF  # noqa: N812
from timm.models import create_model  # pyright: ignore[reportPrivateImportUsage]
from torch import Tensor, nn
from torch.hub import download_url_to_file, get_dir

from traiNNer.archs.arch_util import load_pretrained_network

DEFAULT_CACHE_DIR = os.path.join(get_dir(), "pyiqa")


def load_file_from_url(
    url: str,
    model_dir: str | None = None,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    model_dir = model_dir or DEFAULT_CACHE_DIR

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def dist_to_mos(dist_score: torch.Tensor) -> torch.Tensor:
    """
    Convert distribution prediction to MOS score.
    For datasets with detailed score labels, such as AVA.

    Args:
        dist_score (torch.Tensor): (*, C), C is the class number.

    Returns:
        torch.Tensor: (*, 1) MOS score.
    """
    num_classes = dist_score.shape[-1]
    mos_score = dist_score * torch.arange(1, num_classes + 1).to(dist_score)
    mos_score = mos_score.sum(dim=-1, keepdim=True)
    return mos_score


default_model_urls = {
    "cfanet_fr_kadid_res50": "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/cfanet_fr_kadid_res50-2c4cc61d.pth",
    "cfanet_nr_koniq_res50": "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/cfanet_nr_koniq_res50-9a73138b.pth",
}


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str) -> Callable[..., Tensor]:
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, src: Tensor) -> Tensor:
        src2 = self.norm1(src)
        q = k = src2
        src2, _ = self.self_attn(q, k, value=src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        memory = self.norm2(memory)
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.multihead_attn(query=tgt2, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src: Tensor) -> Tensor:
        output = src

        for layer in self.layers:
            output = layer(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        output = tgt

        for layer in self.layers:
            output = layer(output, memory)

        return output


class GatedConv(nn.Module):
    def __init__(self, weightdim: int, ksz: int = 3) -> None:
        super().__init__()

        self.splitconv = nn.Conv2d(weightdim, weightdim * 2, 1, 1, 0)
        self.act = nn.GELU()

        self.weight_blk = nn.Sequential(
            nn.Conv2d(weightdim, 64, 1, stride=1),
            nn.GELU(),
            nn.Conv2d(64, 64, ksz, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, ksz, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.splitconv(x).chunk(2, dim=1)
        weight = self.weight_blk(x2)
        x1 = self.act(x1)
        return x1 * weight


class CFANet(nn.Module):
    def __init__(
        self,
        semantic_model_name: str = "resnet50",
        model_name: str = "cfanet_fr_kadid_res50",
        backbone_pretrain: bool = True,
        use_ref: bool = True,
        num_class: int = 1,
        inter_dim: int = 256,
        num_heads: int = 4,
        num_attn_layers: int = 1,
        dprate: float = 0.1,
        activation: str = "gelu",
        pretrained: bool = True,
        out_act: bool = False,
        test_img_size: list[int] | None = None,
        default_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        default_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.semantic_model_name = semantic_model_name
        self.semantic_level = -1
        self.use_ref = use_ref

        self.num_class = num_class
        self.test_img_size = test_img_size

        # =============================================================
        # define semantic backbone network
        # =============================================================
        self.semantic_model = create_model(
            semantic_model_name, pretrained=backbone_pretrain, features_only=True
        )
        feature_dim_list = self.semantic_model.feature_info.channels()  # pyright: ignore[reportCallIssue,reportAttributeAccessIssue]
        _feature_dim = feature_dim_list[self.semantic_level]
        _all_feature_dim = sum(feature_dim_list)
        self.fix_bn(self.semantic_model)

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        # =============================================================
        # define self-attention and cross scale attention blocks
        # =============================================================

        self.fusion_mul = 3 if use_ref else 1
        ca_layers = sa_layers = num_attn_layers

        self.act_layer = nn.GELU() if activation == "gelu" else nn.ReLU()
        dim_feedforward = min(4 * inter_dim, 2048)

        # gated local pooling and self-attention
        tmp_layer = TransformerEncoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation,
        )
        self.sa_attn_blks = nn.ModuleList()
        self.dim_reduce = nn.ModuleList()
        self.weight_pool = nn.ModuleList()
        for _idx, dim in enumerate(feature_dim_list):
            dim = dim * 3 if use_ref else dim  # noqa: PLW2901
            if use_ref:
                self.weight_pool.append(
                    nn.Sequential(
                        nn.Conv2d(dim // 3, 64, 1, stride=1),
                        self.act_layer,
                        nn.Conv2d(64, 64, 3, stride=1, padding=1),
                        self.act_layer,
                        nn.Conv2d(64, 1, 3, stride=1, padding=1),
                        nn.Sigmoid(),
                    )
                )
            else:
                self.weight_pool.append(GatedConv(dim))

            self.dim_reduce.append(
                nn.Sequential(
                    nn.Conv2d(dim, inter_dim, 1, 1),
                    self.act_layer,
                )
            )

            self.sa_attn_blks.append(TransformerEncoder(tmp_layer, sa_layers))

        # cross scale attention
        self.attn_blks = nn.ModuleList()
        tmp_layer = TransformerDecoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation,
        )
        for _i in range(len(feature_dim_list) - 1):
            self.attn_blks.append(TransformerDecoder(tmp_layer, ca_layers))

        # attention pooling and MLP layers
        self.attn_pool = TransformerEncoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation,
        )

        linear_dim = inter_dim
        score_linear: list[nn.Module] = [
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.Linear(linear_dim, self.num_class),
        ]

        # make sure output is positive, useful for 2AFC datasets with probability labels
        if out_act and self.num_class == 1:
            score_linear.append(nn.Softplus())

        if self.num_class > 1:
            score_linear.append(nn.Softmax(dim=-1))

        self.score_linear = nn.Sequential(*score_linear)

        self.h_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 32, 1))
        self.w_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 1, 32))

        nn.init.trunc_normal_(self.h_emb.data, std=0.02)
        nn.init.trunc_normal_(self.w_emb.data, std=0.02)
        self._init_linear(self.dim_reduce)
        self._init_linear(self.sa_attn_blks)
        self._init_linear(self.attn_blks)
        self._init_linear(self.attn_pool)

        if pretrained:
            load_pretrained_network(
                self, default_model_urls[model_name], True, weight_keys="params"
            )

        self.eps = 1e-8

    def _init_linear(self, m: nn.Module) -> None:
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)

    def preprocess(self, x: Tensor) -> Tensor:
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def fix_bn(self, model: nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def dist_func(self, x: Tensor, y: Tensor, eps: float = 1e-12) -> Tensor:
        return torch.sqrt((x - y) ** 2 + eps)

    def forward_cross_attention(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        # resize image when testing
        if not self.training:
            if self.test_img_size is not None:
                x = TF.resize(x, self.test_img_size, antialias=True)

        x = self.preprocess(x)
        if self.use_ref:
            assert y is not None
            y = self.preprocess(y)

        dist_feat_list = self.semantic_model(x)
        if self.use_ref:
            ref_feat_list = self.semantic_model(y)
        self.fix_bn(self.semantic_model)
        self.semantic_model.eval()

        start_level = 0
        end_level = len(dist_feat_list)

        _b, _c, th, tw = dist_feat_list[end_level - 1].shape
        pos_emb = torch.cat(
            (
                self.h_emb.repeat(1, 1, 1, self.w_emb.shape[3]),
                self.w_emb.repeat(1, 1, self.h_emb.shape[2], 1),
            ),
            dim=1,
        )

        token_feat_list = []
        for i in reversed(range(start_level, end_level)):
            tmp_dist_feat = dist_feat_list[i]

            # gated local pooling
            if self.use_ref:
                tmp_ref_feat = ref_feat_list[i]  # pyright: ignore[reportPossiblyUnboundVariable]
                diff = self.dist_func(tmp_dist_feat, tmp_ref_feat)

                tmp_feat = torch.cat([tmp_dist_feat, tmp_ref_feat, diff], dim=1)
                weight = self.weight_pool[i](diff)
                tmp_feat = tmp_feat * weight
            else:
                tmp_feat = self.weight_pool[i](tmp_dist_feat)

            if tmp_feat.shape[2] > th and tmp_feat.shape[3] > tw:
                tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))

            # self attention
            tmp_pos_emb = F.interpolate(
                pos_emb, size=tmp_feat.shape[2:], mode="bicubic", align_corners=False
            )
            tmp_pos_emb = tmp_pos_emb.flatten(2).permute(2, 0, 1)

            tmp_feat = self.dim_reduce[i](tmp_feat)
            tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)
            tmp_feat = tmp_feat + tmp_pos_emb

            tmp_feat = self.sa_attn_blks[i](tmp_feat)
            token_feat_list.append(tmp_feat)

        # high level -> low level: coarse to fine
        query = token_feat_list[0]
        query_list = [query]
        for i in range(len(token_feat_list) - 1):
            key_value = token_feat_list[i + 1]
            query = self.attn_blks[i](query, key_value)
            query_list.append(query)

        final_feat = self.attn_pool(query)
        out_score = self.score_linear(final_feat.mean(dim=0))

        return out_score

    def forward(
        self,
        x: Tensor,
        y: Tensor | None = None,
        return_mos: bool = True,
        return_dist: bool = False,
    ) -> list[Tensor] | Tensor:
        if self.use_ref:
            assert y is not None, "Please input y when use reference is True."
        else:
            y = None

        score = self.forward_cross_attention(x, y)

        mos = dist_to_mos(score)

        return_list = []
        if return_mos:
            return_list.append(mos)
        if return_dist:
            return_list.append(score)

        if len(return_list) > 1:
            return [t.squeeze() for t in return_list]
        else:
            return return_list[0].squeeze()
