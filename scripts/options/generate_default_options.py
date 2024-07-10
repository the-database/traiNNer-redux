import os
import re
from os import path as osp
from typing import NotRequired, TypedDict


class ArchInfo(TypedDict):
    names: list[str]
    scales: list[int]
    extras: NotRequired[dict[str, str]]
    gt_override: NotRequired[int]
    folder_name_override: NotRequired[str]


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

    template = template.replace("%archname%", f"{arch['names'][0]}")
    template = template.replace("%scale%", f"{default_scale}x")

    if "gt_override" in arch:
        template = re.sub(r"gt_size: \d+", f"gt_size: {arch['gt_override']}", template)

    template = template.replace("%otf1%\n", template_otf1)
    template = template.replace("    %otf2%\n", template_otf2)

    if template_otf1 and template_otf2:
        template = template.replace("%traindatasettype%", "RealESRGANDataset")
        template = template.replace(
            "    # Path to the LR (low res) images in your training dataset.\n", ""
        )
        template = template.replace("    dataroot_lq: datasets/train/dataset1/lr\n", "")
    else:
        template = template.replace("%traindatasettype%", "PairedImageDataset")

    if mssim_only:
        template = re.sub("loss_weight: [0-9.]+", "loss_weight: 0", template)
        template = template.replace("perceptual_weight: 0.03", "perceptual_weight: 0")
        template = template.replace(
            "type: MSSIMLoss\n    loss_weight: 0",
            "type: MSSIMLoss\n    loss_weight: 1.0",
        )

    return template


archs: list[ArchInfo] = [
    {
        "names": ["ESRGAN", "ESRGAN_lite"],
        "scales": ALL_SCALES,
        "extras": {"use_pixel_unshuffle": "true"},
    },
    {"names": ["ATD"], "scales": ALL_SCALES},
    {"names": ["DAT_2"], "scales": ALL_SCALES},
    {"names": ["HAT_L", "HAT_M", "HAT_S"], "scales": ALL_SCALES},
    {"names": ["OmniSR"], "scales": ALL_SCALES},
    {"names": ["PLKSR"], "scales": ALL_SCALES, "gt_override": 192},
    {
        "names": ["RealPLKSR"],
        "scales": ALL_SCALES,
        "extras": {
            "upsampler": "dysample  # dysample (best on even number scales), pixelshuffle"
        },
        "gt_override": 192,
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
    {"names": ["Compact", "UltraCompact", "SuperUltraCompact"], "scales": ALL_SCALES},
    {"names": ["SwinIR_L", "SwinIR_M", "SwinIR_S"], "scales": ALL_SCALES},
    {"names": ["RGT", "RGT_S"], "scales": ALL_SCALES},
    {"names": ["DRCT", "DRCT_L", "DRCT_XL"], "scales": ALL_SCALES},
    {
        "names": ["SPANPlus", "SPANPlus_STS", "SPANPlus_S", "SPANPlus_ST"],
        "scales": ALL_SCALES,
    },
    {
        "names": ["HiT_SRF", "HiT_SNG", "HiT_SIR"],
        "folder_name_override": "HiT-SR",
        "scales": ALL_SCALES,
    },
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

    os.makedirs(train_folder_path, exist_ok=True)
    os.makedirs(test_folder_path, exist_ok=True)

    template_path_paired = osp.normpath(
        osp.join(__file__, osp.pardir, "./train_default_options_paired.yml")
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

    with (
        open(template_path_paired) as fp,
        open(template_path_otf1) as fo1,
        open(template_path_otf2) as fo2,
        open(template_path_otfbicubic1) as fob1,
        open(template_path_otfbicubic2) as fob2,
        open(template_path_single) as fts,
    ):
        template_paired = fp.read()
        template_otf1 = fo1.read()
        template_otf2 = fo2.read()
        template_otfbicubic1 = fob1.read()
        template_otfbicubic2 = fob2.read()
        template_test_single = fts.read()

        with open(osp.join(train_folder_path, f"{folder_name}.yml"), mode="w") as fw:
            fw.write(final_template(template_paired, arch))

        with open(
            osp.join(train_folder_path, f"{folder_name}_OTF.yml"), mode="w"
        ) as fw:
            fw.write(
                final_template(
                    template_paired, arch, template_otf1, template_otf2, "OTF"
                )
            )

        with open(
            osp.join(train_folder_path, f"{folder_name}_OTF_bicubic_mssim.yml"),
            mode="w",
        ) as fw:
            fw.write(
                final_template(
                    template_paired,
                    arch,
                    template_otfbicubic1,
                    template_otfbicubic2,
                    "OTF_bicubic_mssim",
                    True,
                )
            )

        with open(osp.join(test_folder_path, f"{folder_name}.yml"), mode="w") as fw:
            fw.write(final_template(template_test_single, arch))
