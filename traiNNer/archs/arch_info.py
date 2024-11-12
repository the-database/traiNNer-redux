# TODO refactor
from typing import Any, Literal

ARCHS_WITHOUT_FP16 = {
    "atd",
    "atd_light",
    "cfsr",
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
    "hma",
    "rgt",
    "rgt_s",
    "srformer",
    "srformer_light",
    "swinir",
    "swinir_l",
    "swinir_m",
    "swinir_s",
    "swin2sr",
    "swin2sr_l",
    "swin2sr_m",
    "swin2sr_s",
    "swin2mose",
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
    "cfsr": {
        2: {"psnr": 32.28, "ssim": 0.9300, "dists": 0},
        3: {"psnr": 28.29, "ssim": 0.8553, "dists": 0},
        4: {"psnr": 26.21, "ssim": 0.7897, "dists": 0},
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
        # 2 pixel unshuffle: 33.08   0.9387    (batch 32 lqcrop128 1m iter)
        # 2 not pixel unshuffle: 33.41   0.9407    (batch 16 lqcrop64 1m iter)
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
    "swin2sr_m": {
        2: {"psnr": 33.89, "ssim": 0.9431, "dists": 0},
        4: {"psnr": 27.51, "ssim": 0.8271, "dists": 0},
    },
    "swin2sr_s": {
        2: {"psnr": 32.85, "ssim": 0.9349, "dists": 0},
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

OFFICIAL_SETTINGS_FROMSCRATCH: dict[str, dict[str, Any]] = {
    "atd": {
        "milestones": [250000],
        "total_iter": 300000,
        "warmup_iter": 10000,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 4,
        "accum_iter": "1  # paper: 8",
    },
    "atd_light": {
        "milestones": [250000, 400000, 450000, 475000, 490000],
        "total_iter": 500000,
        "warmup_iter": 20000,
        "lr": "!!float 5e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 2",
    },
    "compact": {
        "milestones": [100000, 200000, 300000, 400000, 425000],
        "total_iter": 450000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 96,
        "batch_size_per_gpu": "16  # recommended: 64",
        "accum_iter": 1,
    },
    "dat": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "rgt": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "plksr": {
        "milestones": [100000, 200000, 300000, 400000, 425000],
        "total_iter": 450000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 96,
        "batch_size_per_gpu": "16  # paper: 64",
        "accum_iter": "1",
    },
    "span": {
        "milestones": [200000, 400000, 600000, 800000],
        "total_iter": 1000000,
        "warmup_iter": -1,
        "lr": "!!float 5e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "16  # paper: 64",
        "accum_iter": "1",
    },
    "esrgan": {
        "milestones": [200000, 400000, 600000, 800000],
        "total_iter": 1000000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": "64  # paper: 32",
        "batch_size_per_gpu": "8",
        "accum_iter": "1  # paper: 2",
    },
    "omnisr": {
        "milestones": [200000, 400000, 600000],
        "total_iter": 800000,
        "warmup_iter": -1,
        "lr": "!!float 5e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "8  # paper: 64",
        "accum_iter": "1",
    },
    "man": {
        "milestones": [800000, 1200000, 140000, 1500000],
        "total_iter": 1600000,
        "warmup_iter": -1,
        "lr": "!!float 5e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "drct": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "drct_l": {
        "milestones": [300000, 500000, 650000, 700000, 750000],
        "total_iter": 800000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "hit_srf": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 5e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "hat_l": {
        "milestones": [300000, 500000, 650000, 700000, 750000],
        "total_iter": 800000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 4,
        "accum_iter": "1  # paper: 8",
    },
    "hat_s": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 4,
        "accum_iter": "1  # paper: 8",
    },
    "srformer": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 4",
    },
    "swinir_m": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "swinir_s": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 4",
    },
    "swin2sr_m": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "swin2sr_s": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 4",
    },
    "": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1",
    },
}

OFFICIAL_SETTINGS_FINETUNE: dict[str, dict[str, Any]] = {
    "atd": {
        "milestones": [150000, 200000, 225000, 240000],
        "total_iter": 250000,
        "warmup_iter": 10000,
        "lr": "!!float 2e-4",
        "lq_size": 96,
        "batch_size_per_gpu": 4,
        "accum_iter": "1  # paper: 8",
    },
    "atd_light": {
        "milestones": [250000, 400000, 450000, 475000, 490000],
        "total_iter": 500000,
        "warmup_iter": 10000,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 2",
    },
    "dat": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "rgt": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "plksr": {
        "milestones": [100000],
        "total_iter": 50000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 96,
        "batch_size_per_gpu": "16  # paper: 64",
        "accum_iter": "1",
    },
    "span": {
        "milestones": [100000, 200000, 300000, 400000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4  # paper: !!float 2.5e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "16  # paper: 64",
        "accum_iter": "1",
    },
    "esrgan": {
        "milestones": [100000, 200000, 300000, 400000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": "64  # paper: 32",
        "batch_size_per_gpu": "8",
        "accum_iter": "1  # paper: 2",
    },
    "omnisr": {
        "milestones": [100000, 200000, 300000],
        "total_iter": 400000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "8  # paper: 64",
        "accum_iter": "1",
    },
    "man": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "drct": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "drct_l": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "hit_srf": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "hat_l": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 4,
        "accum_iter": "1  # paper: 8",
    },
    "hat_s": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 4,
        "accum_iter": "1  # paper: 8",
    },
    "srformer": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 4",
    },
    "swinir_m": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "swinir_s": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 4",
    },
    "swin2sr_m": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "swin2sr_s": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 4",
    },
    "compact": {
        "milestones": [50000, 100000, 150000, 200000, 225000],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 96,
        "batch_size_per_gpu": "8",
        "accum_iter": 1,
    },
    "": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1",
    },
}


def initialize_official_settings(settings: dict[str, dict[str, Any]]) -> None:
    settings["ultracompact"] = settings["compact"]
    settings["superultracompact"] = settings["compact"]

    settings["dat_2"] = settings["dat"]
    settings["dat_s"] = settings["dat"]
    settings["dat_light"] = settings["dat"]

    settings["drct_xl"] = settings["drct_l"]

    settings["esrgan_lite"] = settings["esrgan"]

    settings["hat_m"] = settings["hat_l"]

    settings["hit_sng"] = settings["hit_srf"]
    settings["hit_sir"] = settings["hit_srf"]

    settings["man_light"] = settings["man"]
    settings["man_tiny"] = settings["man"]

    settings["plksr_tiny"] = settings["plksr"]
    settings["realplksr"] = settings["plksr"]

    settings["rgt_s"] = settings["rgt"]

    settings["spanplus"] = settings["span"]
    settings["spanplus_sts"] = settings["span"]
    settings["spanplus_st"] = settings["span"]
    settings["spanplus_s"] = settings["span"]

    settings["srformer_light"] = settings["srformer"]

    settings["swin2sr_l"] = settings["swin2sr_m"]

    settings["swinir_l"] = settings["swinir_m"]


initialize_official_settings(OFFICIAL_SETTINGS_FROMSCRATCH)
initialize_official_settings(OFFICIAL_SETTINGS_FINETUNE)
