from spandrel_extra_arches.architectures.SRFormer import SRFormer
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def srformer(
    in_chans=3,
    img_size=48,
    window_size=24,
    img_range=1.0,
    depths=(6, 6, 6, 6, 6, 6),
    embed_dim=180,
    num_heads=(6, 6, 6, 6, 6, 6),
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv",
    **kwargs,
):
    return SRFormer(
        upscale=Config.get_scale(),
        in_chans=in_chans,
        img_size=img_size,
        window_size=window_size,
        img_range=img_range,
        depths=depths,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        upsampler=upsampler,
        resi_connection=resi_connection,
        **kwargs,
    )


@SPANDREL_REGISTRY.register()
def srformer_light(**kwargs):
    return SRFormer(
        upscale=Config.get_scale(),
        in_chans=3,
        img_size=64,
        window_size=16,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
        **kwargs,
    )
