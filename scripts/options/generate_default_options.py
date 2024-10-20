import os
import re
from os import path as osp
from typing import NotRequired, TypedDict

from traiNNer.archs.arch_info import ARCHS_WITHOUT_FP16


class ArchInfo(TypedDict):
    names: list[str]
    scales: list[int]
    extras: NotRequired[dict[str, str]]
    folder_name_override: NotRequired[str]
    video_override: NotRequired[bool]
    pth_override: NotRequired[bool]
    overrides: NotRequired[dict[str, str]]


ALL_SCALES = [1, 2, 3, 4, 8]


def final_template(
    template: str,
    arch: ArchInfo,
    template_otf1: str = "",
    template_otf2: str = "",
    name_suffix: str = "",
    mssim_only: bool = False,
) -> str:
    default_scale = 4

    template = template.replace(
        "scale: %scale%",
        f"scale: {default_scale}  # {', '.join([str(x) for x in arch['scales']])}",
    )

    arch_type_str = f"type: {arch['names'][0]}"
    if len(arch["names"]) > 1:
        arch_type_str += f"  # {', '.join([str(x) for x in arch['names']])}"

    if "extras" in arch:
        for k, v in arch["extras"].items():
            arch_type_str += f"\n  {k}: {v}"

    if name_suffix:
        template = template.replace(
            "name: %scale%_%archname%", f"name: %scale%_%archname%_{name_suffix}"
        )

    template = template.replace(
        "type: %archname%",
        arch_type_str,
    )

    if arch["names"][0].lower() in ARCHS_WITHOUT_FP16:
        template = template.replace("amp_bf16: false", "amp_bf16: true")

    template = template.replace("%archname%", f"{arch['names'][0]}")
    template = template.replace("%scale%", f"{default_scale}x")

    if "overrides" in arch:
        for k, v in arch["overrides"].items():
            template = re.sub(rf"{k}: .+", rf"{k}: {v}", template)

    if "pth_override" in arch:
        template = template.replace(
            "save_checkpoint_format: safetensors", "save_checkpoint_format: pth"
        )

    template = template.replace("%otf1%\n", template_otf1)
    template = template.replace("    %otf2%\n", template_otf2)

    if template_otf1 and template_otf2:
        template = template.replace("%traindatasettype%", "RealESRGANDataset")
        template = template.replace(
            "    # Path to the LR (low res) images in your training dataset.\n", ""
        )
        template = template.replace(
            "    dataroot_lq: [\n      datasets/train/dataset1/lr,\n      datasets/train/dataset1/lr2\n    ]\n",
            "",
        )
    else:
        template = template.replace("%traindatasettype%", "PairedImageDataset")

    if "video_override" in arch:
        template = template.replace(
            "type: PairedImageDataset", "type: PairedVideoDataset"
        )

        template = template.replace(
            "type: SingleImageDataset", "type: SingleVideoDataset"
        )

        template = re.sub(
            r"(\s+)(dataroot_lq:\s*\[\s*[^]]*])", r"\1\2\1clip_size: 5", template
        )

    if mssim_only:
        template = re.sub("loss_weight: [0-9.]+", "loss_weight: 0", template)
        template = template.replace("perceptual_weight: 0.03", "perceptual_weight: 0")
        template = template.replace(
            "type: MSSSIML1Loss\n    loss_weight: 0",
            "type: MSSSIML1Loss\n    loss_weight: 1.0",
        )

    return template


archs: list[ArchInfo] = [
    {
        "names": ["ESRGAN", "ESRGAN_lite"],
        "scales": ALL_SCALES,
    },
    {"names": ["ATD"], "scales": ALL_SCALES},
    {"names": ["DAT_2"], "scales": ALL_SCALES},
    {"names": ["HAT_L", "HAT_M", "HAT_S"], "scales": ALL_SCALES},
    {"names": ["OmniSR"], "scales": ALL_SCALES},
    {
        "names": ["PLKSR", "PLKSR_Tiny"],
        "scales": ALL_SCALES,
        "overrides": {
            "lq_size": "48  # During training, a square of this size is cropped from LR images. Larger is usually better but uses more VRAM. Previously gt_size, use lq_size = gt_size / scale to convert. Use multiple of 8 for best performance with AMP. (Official default: 96)"
        },
    },
    {
        "names": ["RealPLKSR"],
        "scales": ALL_SCALES,
        "extras": {
            "upsampler": "pixelshuffle  # pixelshuffle, dysample (better quality on even number scales, but does not support dynamic ONNX)",
            "layer_norm": "true  # better quality, not compatible with older models",
        },
        "overrides": {
            "lq_size": "48  # During training, a square of this size is cropped from LR images. Larger is usually better but uses more VRAM. Previously gt_size, use lq_size = gt_size / scale to convert. Use multiple of 8 for best performance with AMP. (Official default: 96)"
        },
    },
    {
        "names": ["RealCUGAN"],
        "scales": [2, 3, 4],
        "extras": {"pro": "true", "fast": "false"},
    },
    {
        "names": ["SPAN"],
        "scales": ALL_SCALES,
        "extras": {"norm": "false  # some pretrains require norm: true"},
    },
    {"names": ["SRFormer", "SRFormer_light"], "scales": ALL_SCALES},
    {
        "names": ["Compact", "UltraCompact", "SuperUltraCompact"],
        "scales": ALL_SCALES,
    },
    {"names": ["SwinIR_L", "SwinIR_M", "SwinIR_S"], "scales": ALL_SCALES},
    {"names": ["RGT", "RGT_S"], "scales": ALL_SCALES},
    {"names": ["DRCT", "DRCT_L", "DRCT_XL"], "scales": ALL_SCALES},
    {
        "names": ["SPANPlus", "SPANPlus_STS", "SPANPlus_S", "SPANPlus_ST"],
        "scales": ALL_SCALES,
        "pth_override": True,
    },
    {
        "names": ["HiT_SRF", "HiT_SNG", "HiT_SIR"],
        "folder_name_override": "HiT-SR",
        "scales": ALL_SCALES,
    },
    {
        "names": ["TSCUNet"],
        "scales": [1, 2, 4, 8],
        "pth_override": True,
        "video_override": True,
    },
    {
        "names": ["SCUNet_aaf6aa"],
        "scales": [1, 2, 4, 8],
        "pth_override": True,
        "folder_name_override": "SCUNet_aaf6aa",
    },
    {
        "names": ["ArtCNN"],
        "scales": ALL_SCALES,
        "extras": {"filters": str(96), "n_block": str(15), "kernel_size": str(3)},
    },
    {
        "names": ["MoSR", "MoSR_T"],
        "scales": ALL_SCALES,
        "extras": {
            "upsampler": "geoensemblepixelshuffle  # geoensemblepixelshuffle, dysample (best on even number scales, does not support dynamic ONNX), pixelshuffle",
            "drop_path": "0  # 0.05",
        },
    },
    {"names": ["LMLT_Base", "LMLT_Large", "LMLT_Tiny"], "scales": ALL_SCALES},
    {
        "names": ["EIMN_L", "EIMN_A"],
        "scales": ALL_SCALES,
        "folder_name_override": "EIMN",
    },
    {"names": ["MAN", "MAN_tiny", "MAN_light"], "scales": ALL_SCALES},
    {
        "names": ["FlexNet", "MetaFlexNet"],
        "scales": ALL_SCALES,
        "extras": {
            "upsampler": "pixelshuffle  # pixelshuffle, nearest+conv, dysample (best on even number scales, does not support dynamic ONNX)"
        },
    },
    {"names": ["HMA"], "scales": ALL_SCALES},
]

for arch in archs:
    folder_name = arch["names"][0].split("_")[0]

    if "folder_name_override" in arch:
        folder_name = arch["folder_name_override"]

    train_folder_path = osp.normpath(
        osp.join(
            __file__, osp.pardir, osp.pardir, osp.pardir, "./options/train", folder_name
        )
    )
    test_folder_path = osp.normpath(
        osp.join(
            __file__, osp.pardir, osp.pardir, osp.pardir, "./options/test", folder_name
        )
    )
    onnx_folder_path = osp.normpath(
        osp.join(
            __file__, osp.pardir, osp.pardir, osp.pardir, "./options/onnx", folder_name
        )
    )

    os.makedirs(train_folder_path, exist_ok=True)
    os.makedirs(test_folder_path, exist_ok=True)
    os.makedirs(onnx_folder_path, exist_ok=True)

    template_path_paired_fromscratch = osp.normpath(
        osp.join(__file__, osp.pardir, "./train_default_options_paired_fromscratch.yml")
    )

    template_path_paired_finetune = osp.normpath(
        osp.join(__file__, osp.pardir, "./train_default_options_paired_finetune.yml")
    )

    template_path_otf1 = osp.normpath(
        osp.join(__file__, osp.pardir, "./train_default_options_otf1.yml")
    )

    template_path_otf2 = osp.normpath(
        osp.join(__file__, osp.pardir, "./train_default_options_otf2.yml")
    )

    template_path_otfbicubic1 = osp.normpath(
        osp.join(__file__, osp.pardir, "./train_default_options_otfbicubic1.yml")
    )

    template_path_otfbicubic2 = osp.normpath(
        osp.join(__file__, osp.pardir, "./train_default_options_otfbicubic2.yml")
    )

    template_path_single = osp.normpath(
        osp.join(__file__, osp.pardir, "./test_default_options_single.yml")
    )

    template_path_onnx = osp.normpath(
        osp.join(__file__, osp.pardir, "./onnx_default_options.yml")
    )

    with (
        open(template_path_paired_fromscratch) as fps,
        open(template_path_paired_finetune) as fpf,
        open(template_path_otf1) as fo1,
        open(template_path_otf2) as fo2,
        open(template_path_otfbicubic1) as fob1,
        open(template_path_otfbicubic2) as fob2,
        open(template_path_single) as fts,
        open(template_path_onnx) as fox,
    ):
        template_paired_fromscratch = fps.read()
        template_paired_finetune = fpf.read()
        template_otf1 = fo1.read()
        template_otf2 = fo2.read()
        template_otfbicubic1 = fob1.read()
        template_otfbicubic2 = fob2.read()
        template_test_single = fts.read()
        template_onnx = fox.read()

        with open(
            osp.join(train_folder_path, f"{folder_name}_fromscratch.yml"), mode="w"
        ) as fw:
            fw.write(final_template(template_paired_fromscratch, arch))

        with open(
            osp.join(train_folder_path, f"{folder_name}_finetune.yml"), mode="w"
        ) as fw:
            fw.write(final_template(template_paired_finetune, arch))

        with open(
            osp.join(train_folder_path, f"{folder_name}_OTF_fromscratch.yml"), mode="w"
        ) as fw:
            fw.write(
                final_template(
                    template_paired_fromscratch,
                    arch,
                    template_otf1,
                    template_otf2,
                    "OTF_fromscratch",
                )
            )

        with open(
            osp.join(train_folder_path, f"{folder_name}_OTF_finetune.yml"), mode="w"
        ) as fw:
            fw.write(
                final_template(
                    template_paired_finetune,
                    arch,
                    template_otf1,
                    template_otf2,
                    "OTF_finetune",
                )
            )

        with open(
            osp.join(
                train_folder_path,
                f"{folder_name}_OTF_bicubic_ms_ssim_l1_fromscratch.yml",
            ),
            mode="w",
        ) as fw:
            fw.write(
                final_template(
                    template_paired_fromscratch,
                    arch,
                    template_otfbicubic1,
                    template_otfbicubic2,
                    "OTF_bicubic_ms_ssim_l1",
                    True,
                )
            )

        with open(osp.join(test_folder_path, f"{folder_name}.yml"), mode="w") as fw:
            fw.write(final_template(template_test_single, arch))

        with open(osp.join(onnx_folder_path, f"{folder_name}.yml"), mode="w") as fw:
            fw.write(final_template(template_onnx, arch, template_otf1, template_otf2))
