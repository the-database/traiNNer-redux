import os
import shutil
from os import path as osp
from typing import TypedDict


class ArchInfo(TypedDict):
    names: list[str]
    scales: list[int]


ALL_SCALES = [1, 2, 3, 4, 8]
SCALES_234 = [2, 3, 4]


archs: list[ArchInfo] = [
    {
        "names": ["ESRGAN", "ESRGAN_lite"],
        "scales": ALL_SCALES,
    },
    {"names": ["ATD"], "scales": ALL_SCALES},
    {"names": ["DAT_2"], "scales": SCALES_234},
    {"names": ["HAT_L", "HAT_M", "HAT_S"], "scales": ALL_SCALES},
    {"names": ["OmniSR"], "scales": SCALES_234},
    {"names": ["PLKSR", "RealPLKSR"], "scales": SCALES_234},
    {"names": ["RealCUGAN"], "scales": SCALES_234},
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

    for name in arch["names"]:
        for scale in arch["scales"]:
            shutil.copy(
                osp.normpath(osp.join(__file__, osp.pardir, "./default_options_paired.yml")),
                osp.join(folder_path, f"{scale}x_{name}.yml"),
            )

            shutil.copy(
                osp.normpath(osp.join(__file__, osp.pardir, "./default_options_paired.yml")),
                osp.join(folder_path, f"{scale}x_{name}_OTF.yml"),
            )
