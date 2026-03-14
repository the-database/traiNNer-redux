from spandrel.architectures.NAFNet import NAFNet

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def nafnet64(
    scale: int = 1,
    img_channel: int = 3,
    width: int = 64,
    enc_blk_nums: list[int] | None = None,
    middle_blk_num: int = 12,
    dec_blk_nums: list[int] | None = None,
) -> NAFNet:
    if dec_blk_nums is None:
        dec_blk_nums = [2, 2, 2, 2]
    if enc_blk_nums is None:
        enc_blk_nums = [2, 2, 4, 8]
    return NAFNet(
        img_channel=img_channel,
        width=width,
        enc_blk_nums=enc_blk_nums,
        middle_blk_num=middle_blk_num,
        dec_blk_nums=dec_blk_nums,
    )


@SPANDREL_REGISTRY.register()
def nafnet32(
    scale: int = 1,
    img_channel: int = 3,
    width: int = 32,
    enc_blk_nums: list[int] | None = None,
    middle_blk_num: int = 12,
    dec_blk_nums: list[int] | None = None,
) -> NAFNet:
    if dec_blk_nums is None:
        dec_blk_nums = [2, 2, 2, 2]
    if enc_blk_nums is None:
        enc_blk_nums = [2, 2, 4, 8]
    return NAFNet(
        img_channel=img_channel,
        width=width,
        enc_blk_nums=enc_blk_nums,
        middle_blk_num=middle_blk_num,
        dec_blk_nums=dec_blk_nums,
    )
