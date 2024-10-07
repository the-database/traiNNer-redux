# TODO refactor
from typing import Literal

ARCHS_WITHOUT_FP16 = {
    "atd",
    "atd_light",
    "dat_2",
    "drct",
    "drct_l",
    "drct_xl",
    "hat_l",
    "hat_m",
    "hat_s",
    "rgt",
    "rgt_s",
    "srformer",
    "srformer_light",
    "swinir_l",
    "swinir_m",
    "swinir_s",
    "hit_srf",
    "hit_sng",
    "hit_sir",
}

# Urban100
OFFICIAL_METRICS: dict[
    str, dict[int, dict[Literal["psnr", "ssim", "dists"], float]]
] = {
    "atd": {
        2: {"psnr": 34.73, "ssim": 0.9476, "dists": 0},
        3: {"psnr": 30.52, "ssim": 0.8924, "dists": 0},
        4: {"psnr": 28.22, "ssim": 0.8414, "dists": 0},
    },
    "dat_2": {
        2: {"psnr": 34.31, "ssim": 0.9457, "dists": 0},
        3: {"psnr": 30.13, "ssim": 0.8878, "dists": 0},
        4: {"psnr": 27.86, "ssim": 0.8341, "dists": 0},
    },
    "drct_l": {
        2: {"psnr": 35.17, "ssim": 0.9516, "dists": 0},
        3: {"psnr": 31.14, "ssim": 0.9004, "dists": 0},
        4: {"psnr": 28.70, "ssim": 0.8508, "dists": 0},
    },
    "hat_l": {
        2: {"psnr": 35.09, "ssim": 0.9513, "dists": 0},
        3: {"psnr": 30.92, "ssim": 0.8981, "dists": 0},
        4: {"psnr": 28.60, "ssim": 0.8498, "dists": 0},
    },
    "plksr": {
        2: {"psnr": 33.36, "ssim": 0.9395, "dists": 0},
        3: {"psnr": 29.10, "ssim": 0.8713, "dists": 0},
        4: {"psnr": 26.85, "ssim": 0.8097, "dists": 0},
    },
    "span": {
        2: {"psnr": 32.24, "ssim": 0.9294, "dists": 0},
        4: {"psnr": 26.18, "ssim": 0.7879, "dists": 0},
    },
    "esrgan": {
        4: {"psnr": 27.03, "ssim": 0.8153, "dists": 0},
    },
}
