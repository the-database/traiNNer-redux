from spandrel.architectures.DAT import DAT
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def dat(**kwargs):
    return DAT(upscale=Config.get_scale(), **kwargs)


@SPANDREL_REGISTRY.register()
def dat_2(**kwargs):
    return DAT(upscale=Config.get_scale(), in_chans=3, img_size=64, img_range=1., split_size=[8, 32],
               depth=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], expansion_factor=2,
               resi_connection='1conv', **kwargs)
