# TODO refactor
from typing import Literal

ARCHS_WITHOUT_FP16 = {
    "atd",
    "atd_light",
    "dat",
    "dat_2",
    "dat_s",
    "dat_light",
    "drct",
    "drct_l",
    "drct_xl",
    "flexnet",
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
    "hit_lmlt",
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
    "atd_light": {
        2: {"psnr": 33.27, "ssim": 0.9375, "dists": 0},
        3: {"psnr": 29.17, "ssim": 0.8709, "dists": 0},
        4: {"psnr": 26.97, "ssim": 0.8107, "dists": 0},
    },
    "dat": {
        2: {"psnr": 34.37, "ssim": 0.9458, "dists": 0},
        3: {"psnr": 30.18, "ssim": 0.8886, "dists": 0},
        4: {"psnr": 27.87, "ssim": 0.8343, "dists": 0},
    },
    "dat_s": {
        2: {"psnr": 34.12, "ssim": 0.9444, "dists": 0},
        3: {"psnr": 29.98, "ssim": 0.8846, "dists": 0},
        4: {"psnr": 27.68, "ssim": 0.8300, "dists": 0},
    },
    "dat_light": {
        2: {"psnr": 32.89, "ssim": 0.9346, "dists": 0},
        3: {"psnr": 28.89, "ssim": 0.8666, "dists": 0},
        4: {"psnr": 26.64, "ssim": 0.8033, "dists": 0},
    },
    "dat_2": {
        2: {"psnr": 34.31, "ssim": 0.9457, "dists": 0},
        3: {"psnr": 30.13, "ssim": 0.8878, "dists": 0},
        4: {"psnr": 27.86, "ssim": 0.8341, "dists": 0},
    },
    "drct": {
        2: {"psnr": 34.54, "ssim": 0.9474, "dists": 0},
        3: {"psnr": 30.34, "ssim": 0.8910, "dists": 0},
        4: {"psnr": 28.06, "ssim": 0.8378, "dists": 0},
    },
    "drct_l": {
        2: {"psnr": 35.17, "ssim": 0.9516, "dists": 0},
        3: {"psnr": 31.14, "ssim": 0.9004, "dists": 0},
        4: {"psnr": 28.70, "ssim": 0.8508, "dists": 0},
    },
    "eimn_a": {
        2: {"psnr": 33.15, "ssim": 0.9373, "dists": 0},
        3: {"psnr": 28.87, "ssim": 0.8660, "dists": 0},
        4: {"psnr": 26.68, "ssim": 0.8027, "dists": 0},
    },
    "eimn_l": {
        2: {"psnr": 33.23, "ssim": 0.9381, "dists": 0},
        3: {"psnr": 29.05, "ssim": 0.8698, "dists": 0},
        4: {"psnr": 26.88, "ssim": 0.8084, "dists": 0},
    },
    "esrgan": {
        4: {"psnr": 27.03, "ssim": 0.8153, "dists": 0},
    },
    "hat_s": {
        2: {"psnr": 34.31, "ssim": 0.9459, "dists": 0},
        3: {"psnr": 30.15, "ssim": 0.8879, "dists": 0},
        4: {"psnr": 27.87, "ssim": 0.8346, "dists": 0},
    },
    "hat_m": {
        2: {"psnr": 34.45, "ssim": 0.9466, "dists": 0},
        3: {"psnr": 30.23, "ssim": 0.8896, "dists": 0},
        4: {"psnr": 27.97, "ssim": 0.8368, "dists": 0},
    },
    "hat_l": {
        2: {"psnr": 35.09, "ssim": 0.9513, "dists": 0},
        3: {"psnr": 30.92, "ssim": 0.8981, "dists": 0},
        4: {"psnr": 28.60, "ssim": 0.8498, "dists": 0},
    },
    "man": {
        2: {"psnr": 33.73, "ssim": 0.9422, "dists": 0},
        3: {"psnr": 29.52, "ssim": 0.8782, "dists": 0},
        4: {"psnr": 27.26, "ssim": 0.8197, "dists": 0},
    },
    "man_tiny": {
        4: {"psnr": 25.84, "ssim": 0.7786, "dists": 0},
    },
    "man_light": {
        4: {"psnr": 26.70, "ssim": 0.8052, "dists": 0},
    },
    "omnisr": {
        2: {"psnr": 33.30, "ssim": 0.9386, "dists": 0},
        3: {"psnr": 29.12, "ssim": 0.8712, "dists": 0},
        4: {"psnr": 26.95, "ssim": 0.8105, "dists": 0},
    },
    "plksr": {
        2: {"psnr": 33.36, "ssim": 0.9395, "dists": 0},
        3: {"psnr": 29.10, "ssim": 0.8713, "dists": 0},
        4: {"psnr": 26.85, "ssim": 0.8097, "dists": 0},
    },
    "plksr_tiny": {
        2: {"psnr": 32.58, "ssim": 0.9328, "dists": 0},
        3: {"psnr": 28.51, "ssim": 0.8599, "dists": 0},
        4: {"psnr": 26.34, "ssim": 0.7942, "dists": 0},
    },
    "rgt": {
        2: {"psnr": 34.47, "ssim": 0.9467, "dists": 0},
        3: {"psnr": 30.28, "ssim": 0.8899, "dists": 0},
        4: {"psnr": 27.98, "ssim": 0.8369, "dists": 0},
    },
    "rgt_s": {
        2: {"psnr": 34.32, "ssim": 0.9457, "dists": 0},
        3: {"psnr": 30.18, "ssim": 0.8884, "dists": 0},
        4: {"psnr": 27.89, "ssim": 0.8347, "dists": 0},
    },
    "span": {
        2: {"psnr": 32.24, "ssim": 0.9294, "dists": 0},
        4: {"psnr": 26.18, "ssim": 0.7879, "dists": 0},
    },
    "srformer": {
        2: {"psnr": 34.09, "ssim": 0.9449, "dists": 0},
        3: {"psnr": 30.04, "ssim": 0.8865, "dists": 0},
        4: {"psnr": 27.68, "ssim": 0.8311, "dists": 0},
    },
    "srformer_light": {
        2: {"psnr": 32.91, "ssim": 0.9353, "dists": 0},
        3: {"psnr": 28.81, "ssim": 0.8655, "dists": 0},
        4: {"psnr": 26.67, "ssim": 0.8032, "dists": 0},
    },
    "swinir_s": {
        2: {"psnr": 32.76, "ssim": 0.9340, "dists": 0},
        3: {"psnr": 28.66, "ssim": 0.8624, "dists": 0},
        4: {"psnr": 26.47, "ssim": 0.7980, "dists": 0},
    },
    "swinir_m": {
        2: {"psnr": 33.81, "ssim": 0.9427, "dists": 0},
        3: {"psnr": 29.75, "ssim": 0.8826, "dists": 0},
        4: {"psnr": 27.45, "ssim": 0.8254, "dists": 0},
    },
    "hit_srf": {
        2: {"psnr": 33.13, "ssim": 0.9372, "dists": 0},
        3: {"psnr": 28.99, "ssim": 0.8687, "dists": 0},
        4: {"psnr": 26.80, "ssim": 0.8069, "dists": 0},
    },
    "hit_sng": {
        2: {"psnr": 33.01, "ssim": 0.9360, "dists": 0},
        3: {"psnr": 28.91, "ssim": 0.8671, "dists": 0},
        4: {"psnr": 26.75, "ssim": 0.8053, "dists": 0},
    },
    "hit_sir": {
        2: {"psnr": 33.02, "ssim": 0.9365, "dists": 0},
        3: {"psnr": 28.93, "ssim": 0.8673, "dists": 0},
        4: {"psnr": 26.71, "ssim": 0.8045, "dists": 0},
    },
    "lmlt_base": {
        2: {"psnr": 32.52, "ssim": 0.9316, "dists": 0},
        3: {"psnr": 28.48, "ssim": 0.8581, "dists": 0},
        4: {"psnr": 26.44, "ssim": 0.7949, "dists": 0},
    },
    "lmlt_large": {
        2: {"psnr": 32.75, "ssim": 0.9336, "dists": 0},
        3: {"psnr": 28.72, "ssim": 0.8628, "dists": 0},
        4: {"psnr": 26.63, "ssim": 0.8001, "dists": 0},
    },
    "lmlt_tiny": {
        2: {"psnr": 32.04, "ssim": 0.9273, "dists": 0},
        3: {"psnr": 28.10, "ssim": 0.8503, "dists": 0},
        4: {"psnr": 26.08, "ssim": 0.7838, "dists": 0},
    },
}
