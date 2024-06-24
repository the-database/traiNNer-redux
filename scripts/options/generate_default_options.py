import os
from os import path as osp
from typing import NotRequired, TypedDict


class ArchInfo(TypedDict):
    names: list[str]
    scales: list[int]
    extras: NotRequired[dict[str, str]]


ALL_SCALES = [1, 2, 3, 4, 8]
SCALES_234 = [2, 3, 4]


def final_template(template: str, arch: ArchInfo) -> str:

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

    template = template.replace(
        "type: %archname%",
        arch_type_str,
    )

    template = template.replace("%archname%", f"{arch['names'][0]}")
    template = template.replace("%scale%", f"{default_scale}x")
    return template


archs: list[ArchInfo] = [
    {
        "names": ["ESRGAN", "ESRGAN_lite"],
        "scales": ALL_SCALES,
    },
    {"names": ["ATD"], "scales": ALL_SCALES},
    {"names": ["DAT_2"], "scales": SCALES_234},
    {"names": ["HAT_L", "HAT_M", "HAT_S"], "scales": ALL_SCALES},
    {"names": ["OmniSR"], "scales": SCALES_234},
    {"names": ["PLKSR"], "scales": SCALES_234},
    {"names": ["RealPLKSR"], "scales": SCALES_234},
    {
        "names": ["RealCUGAN"],
        "scales": SCALES_234,
        "extras": {"pro": "true", "fast": "false"},
    },
    {"names": ["SPAN"], "scales": [2, 4]},
    {"names": ["SRFormer", "SRFormer_light"], "scales": ALL_SCALES},
    {"names": ["Compact", "UltraCompact", "SuperUltraCompact"], "scales": [1, 2, 4]},
    {"names": ["SwinIR_L", "SwinIR_M", "SwinIR_S"], "scales": ALL_SCALES},
]

for arch in archs:
    folder_name = arch["names"][0].split("_")[0]
    folder_path = osp.normpath(
        osp.join(
            __file__, osp.pardir, osp.pardir, osp.pardir, "./options/train", folder_name
        )
    )
    # print(folder_name, folder_path)
    os.makedirs(folder_path, exist_ok=True)

    template_path_paired = osp.normpath(
        osp.join(__file__, osp.pardir, "./default_options_paired.yml")
    )

    template_path_otf = osp.normpath(
        osp.join(__file__, osp.pardir, "./default_options_otf.yml")
    )

    with open(template_path_paired) as fp, open(template_path_otf) as fo:
        template_paired = fp.read()
        template_otf = fo.read()

        with open(osp.join(folder_path, f"{folder_name}.yml"), mode="w") as fw:
            fw.write(final_template(template_paired, arch))

        with open(osp.join(folder_path, f"{folder_name}_OTF.yml"), mode="w") as fw:
            fw.write(final_template(template_otf, arch))
