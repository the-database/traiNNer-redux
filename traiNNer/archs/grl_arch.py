from typing import Literal

from spandrel.architectures.GRL import GRL
from torch import nn

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def grl_b(
    scale: int = 4,
    img_size: int = 64,
    in_channels: int = 3,
    out_channels: int = 3,
    embed_dim: int = 180,
    img_range: float = 1.0,
    upsampler: str = "pixelshuffle",
    depths: list[int] = [4, 4, 8, 8, 8, 4, 4],  # noqa: B006
    num_heads_window: list[int] = [3, 3, 3, 3, 3, 3, 3],  # noqa: B006
    num_heads_stripe: list[int] = [3, 3, 3, 3, 3, 3, 3],  # noqa: B006
    window_size: int = 8,
    stripe_size: list[int] = [  # noqa: B006
        8,
        None,
    ],  # used for stripe window attention # type: ignore
    stripe_groups: list[int | None] = [None, 4],  # noqa: B006
    stripe_shift: bool = True,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qkv_proj_type: str = "linear",
    anchor_proj_type: str = "avgpool",
    anchor_one_stage: bool = True,
    anchor_window_down_factor: int = 4,
    out_proj_type: Literal["linear", "conv2d"] = "linear",
    local_connection: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
    pretrained_window_size: list[int] = [0, 0],  # noqa: B006
    pretrained_stripe_size: list[int] = [0, 0],  # noqa: B006
    conv_type: str = "1conv",
    init_method: str = "n",  # initialization method of the weight parameters used to train large scale models.
    fairscale_checkpoint: bool = False,  # fairscale activation checkpointing
    offload_to_cpu: bool = False,
    euclidean_dist: bool = False,
) -> GRL:
    return GRL(
        upscale=scale,
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        img_range=img_range,
        upsampler=upsampler,
        depths=depths,
        num_heads_window=num_heads_window,
        num_heads_stripe=num_heads_stripe,
        window_size=window_size,
        stripe_size=stripe_size,
        stripe_groups=stripe_groups,
        stripe_shift=stripe_shift,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qkv_proj_type=qkv_proj_type,
        anchor_proj_type=anchor_proj_type,
        anchor_one_stage=anchor_one_stage,
        anchor_window_down_factor=anchor_window_down_factor,
        out_proj_type=out_proj_type,
        local_connection=local_connection,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        pretrained_window_size=pretrained_window_size,
        pretrained_stripe_size=pretrained_stripe_size,
        conv_type=conv_type,
        init_method=init_method,
        fairscale_checkpoint=fairscale_checkpoint,
        offload_to_cpu=offload_to_cpu,
        euclidean_dist=euclidean_dist,
    )


@SPANDREL_REGISTRY.register()
def grl_s(
    scale: int = 4,
    img_size: int = 64,
    in_channels: int = 3,
    out_channels: int = 3,
    embed_dim: int = 128,
    img_range: float = 1.0,
    upsampler: str = "pixelshuffle",
    depths: list[int] = [4, 4, 4, 4],  # noqa: B006
    num_heads_window: list[int] = [2, 2, 2, 2],  # noqa: B006
    num_heads_stripe: list[int] = [2, 2, 2, 2],  # noqa: B006
    window_size: int = 8,
    stripe_size: list[int] = [  # noqa: B006
        8,
        None,
    ],  # used for stripe window attention # type: ignore
    stripe_groups: list[int | None] = [None, 4],  # noqa: B006
    stripe_shift: bool = True,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qkv_proj_type: str = "linear",
    anchor_proj_type: str = "avgpool",
    anchor_one_stage: bool = True,
    anchor_window_down_factor: int = 4,
    out_proj_type: Literal["linear", "conv2d"] = "linear",
    local_connection: bool = False,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
    pretrained_window_size: list[int] = [0, 0],  # noqa: B006
    pretrained_stripe_size: list[int] = [0, 0],  # noqa: B006
    conv_type: str = "1conv",
    init_method: str = "n",  # initialization method of the weight parameters used to train large scale models.
    fairscale_checkpoint: bool = False,  # fairscale activation checkpointing
    offload_to_cpu: bool = False,
    euclidean_dist: bool = False,
) -> GRL:
    return GRL(
        upscale=scale,
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        img_range=img_range,
        upsampler=upsampler,
        depths=depths,
        num_heads_window=num_heads_window,
        num_heads_stripe=num_heads_stripe,
        window_size=window_size,
        stripe_size=stripe_size,
        stripe_groups=stripe_groups,
        stripe_shift=stripe_shift,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qkv_proj_type=qkv_proj_type,
        anchor_proj_type=anchor_proj_type,
        anchor_one_stage=anchor_one_stage,
        anchor_window_down_factor=anchor_window_down_factor,
        out_proj_type=out_proj_type,
        local_connection=local_connection,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        pretrained_window_size=pretrained_window_size,
        pretrained_stripe_size=pretrained_stripe_size,
        conv_type=conv_type,
        init_method=init_method,
        fairscale_checkpoint=fairscale_checkpoint,
        offload_to_cpu=offload_to_cpu,
        euclidean_dist=euclidean_dist,
    )


@SPANDREL_REGISTRY.register()
def grl_t(
    scale: int = 4,
    img_size: int = 64,
    in_channels: int = 3,
    out_channels: int = 3,
    embed_dim: int = 64,
    img_range: float = 1.0,
    upsampler: str = "pixelshuffledirect",
    depths: list[int] = [4, 4, 4, 4],  # noqa: B006
    num_heads_window: list[int] = [2, 2, 2, 2],  # noqa: B006
    num_heads_stripe: list[int] = [2, 2, 2, 2],  # noqa: B006
    window_size: int = 8,
    stripe_size: list[int] = [  # noqa: B006
        8,
        None,
    ],  # used for stripe window attention # type: ignore
    stripe_groups: list[int | None] = [None, 4],  # noqa: B006
    stripe_shift: bool = True,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qkv_proj_type: str = "linear",
    anchor_proj_type: str = "avgpool",
    anchor_one_stage: bool = True,
    anchor_window_down_factor: int = 4,
    out_proj_type: Literal["linear", "conv2d"] = "linear",
    local_connection: bool = False,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
    pretrained_window_size: list[int] = [0, 0],  # noqa: B006
    pretrained_stripe_size: list[int] = [0, 0],  # noqa: B006
    conv_type: str = "1conv",
    init_method: str = "n",  # initialization method of the weight parameters used to train large scale models.
    fairscale_checkpoint: bool = False,  # fairscale activation checkpointing
    offload_to_cpu: bool = False,
    euclidean_dist: bool = False,
) -> GRL:
    return GRL(
        upscale=scale,
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        img_range=img_range,
        upsampler=upsampler,
        depths=depths,
        num_heads_window=num_heads_window,
        num_heads_stripe=num_heads_stripe,
        window_size=window_size,
        stripe_size=stripe_size,
        stripe_groups=stripe_groups,
        stripe_shift=stripe_shift,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qkv_proj_type=qkv_proj_type,
        anchor_proj_type=anchor_proj_type,
        anchor_one_stage=anchor_one_stage,
        anchor_window_down_factor=anchor_window_down_factor,
        out_proj_type=out_proj_type,
        local_connection=local_connection,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        pretrained_window_size=pretrained_window_size,
        pretrained_stripe_size=pretrained_stripe_size,
        conv_type=conv_type,
        init_method=init_method,
        fairscale_checkpoint=fairscale_checkpoint,
        offload_to_cpu=offload_to_cpu,
        euclidean_dist=euclidean_dist,
    )
