from typing import Any, Literal

from msgspec import Struct, field


class StrictStruct(Struct, forbid_unknown_fields=True):
    pass


class SchedulerOptions(StrictStruct):
    type: str
    milestones: list[int]
    gamma: float


class WandbOptions(StrictStruct):
    resume_id: str | None = None
    project: str | None = None


class DatasetOptions(StrictStruct):
    name: str
    type: str
    io_backend: dict[str, Any]

    num_worker_per_gpu: int | None = None
    batch_size_per_gpu: int | None = None
    accum_iter: int = 1

    use_hflip: bool | None = None
    use_rot: bool | None = None
    mean: list[float] | None = None
    std: list[float] | None = None
    gt_size: int | None = None
    lq_size: int | None = None
    color: Literal["y"] | None = None
    phase: str | None = None
    scale: int | None = None
    dataset_enlarge_ratio: Literal["auto"] | int = "auto"
    prefetch_mode: str | None = None
    pin_memory: bool = True
    persistent_workers: bool = True
    num_prefetch_queue: int = 1

    clip_size: int | None = None

    dataroot_gt: str | list[str] | None = None
    dataroot_lq: str | list[str] | None = None
    meta_info: str | None = None
    filename_tmpl: str = "{}"

    blur_kernel_size: int = 12
    kernel_list: list[str] = field(
        default_factory=lambda: [
            "iso",
            "aniso",
            "generalized_iso",
            "generalized_aniso",
            "plateau_iso",
            "plateau_aniso",
        ]
    )
    kernel_prob: list[float] = field(
        default_factory=lambda: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    )
    kernel_range: tuple[int, int] = (5, 17)
    sinc_prob: float = 0
    blur_sigma: tuple[float, float] = (0.2, 2)
    betag_range: tuple[float, float] = (0.5, 4)
    betap_range: tuple[float, float] = (1, 2)

    blur_kernel_size2: int = 12
    kernel_list2: list[str] = field(
        default_factory=lambda: [
            "iso",
            "aniso",
            "generalized_iso",
            "generalized_aniso",
            "plateau_iso",
            "plateau_aniso",
        ]
    )
    kernel_prob2: list[float] = field(
        default_factory=lambda: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    )
    kernel_range2: tuple[int, int] = (5, 17)
    sinc_prob2: float = 0
    blur_sigma2: tuple[float, float] = (0.2, 1)
    betag_range2: tuple[float, float] = (0.5, 4)
    betap_range2: tuple[float, float] = (1, 2)

    final_sinc_prob: float = 0
    final_kernel_range: tuple[int, int] = (5, 17)


class PathOptions(StrictStruct):
    experiments_root: str | None = None
    models: str | None = None
    resume_models: str | None = None
    training_states: str | None = None
    log: str | None = None
    visualization: str | None = None
    results_root: str | None = None

    pretrain_network_g: str | None = None
    pretrain_network_g_path: str | None = None
    param_key_g: str | None = None
    strict_load_g: bool = True
    resume_state: str | None = None
    pretrain_network_g_ema: str | None = None

    pretrain_network_d: str | None = None
    param_key_d: str | None = None
    strict_load_d: bool = True
    ignore_resume_networks: list[str] | None = None


class OnnxOptions(StrictStruct):
    dynamo: bool = False
    opset: int = 20
    use_static_shapes: bool = False
    shape: str = "3x256x256"
    verify: bool = True
    fp16: bool = False
    optimize: bool = True


class TrainOptions(StrictStruct):
    total_iter: int
    optim_g: dict[str, Any]
    ema_decay: float = 0
    grad_clip: bool = False
    warmup_iter: int = -1
    scheduler: SchedulerOptions | None = None
    optim_d: dict[str, Any] | None = None

    pixel_opt: dict[str, Any] | None = None
    mssim_opt: dict[str, Any] | None = None
    ms_ssim_l1_opt: dict[str, Any] | None = None
    perceptual_opt: dict[str, Any] | None = None
    contextual_opt: dict[str, Any] | None = None
    dists_opt: dict[str, Any] | None = None
    hr_inversion_opt: dict[str, Any] | None = None
    dinov2_opt: dict[str, Any] | None = None
    topiq_opt: dict[str, Any] | None = None
    ldl_opt: dict[str, Any] | None = None
    hsluv_opt: dict[str, Any] | None = None
    gan_opt: dict[str, Any] | None = None
    color_opt: dict[str, Any] | None = None
    luma_opt: dict[str, Any] | None = None
    avg_opt: dict[str, Any] | None = None
    bicubic_opt: dict[str, Any] | None = None

    use_moa: bool = False
    moa_augs: list[str] = field(
        default_factory=lambda: ["none", "mixup", "cutmix", "resizemix"]
    )
    moa_probs: list[float] = field(
        default_factory=lambda: [0.4, 0.084, 0.084, 0.084, 0.348]
    )
    moa_debug: bool = False
    moa_debug_limit: int = 100


class ValOptions(StrictStruct):
    val_enabled: bool
    save_img: bool
    val_freq: int | None = None
    suffix: str | None = None

    metrics_enabled: bool = False
    metrics: dict[str, Any] | None = None
    pbar: bool = True


class LogOptions(StrictStruct):
    print_freq: int
    save_checkpoint_freq: int
    use_tb_logger: bool
    save_checkpoint_format: Literal["safetensors", "pth"] = "safetensors"
    wandb: WandbOptions | None = None


class ReduxOptions(StrictStruct):
    name: str
    scale: int
    num_gpu: Literal["auto"] | int
    path: PathOptions

    network_g: dict[str, Any]
    network_d: dict[str, Any] | None = None

    manual_seed: int | None = None
    deterministic: bool | None = None
    dist: bool | None = None
    launcher: str | None = None
    rank: int | None = None
    world_size: int | None = None
    auto_resume: bool | None = None
    resume: int = 0
    is_train: bool | None = None
    root_path: str | None = None

    use_amp: bool = False
    amp_bf16: bool = False
    fast_matmul: bool = False
    detect_anomaly: bool = False

    high_order_degradation: bool = False
    high_order_degradations_debug: bool = False
    high_order_degradations_debug_limit: int = 100

    lq_usm: bool = False
    lq_usm_radius_range: tuple[int, int] = (1, 25)

    blur_prob: float = 0
    resize_prob: list[float] = field(default_factory=lambda: [0.2, 0.7, 0.1])
    resize_mode_list: list[str] = field(
        default_factory=lambda: ["bilinear", "bicubic", "nearest-exact"]
    )
    resize_mode_prob: list[float] = field(
        default_factory=lambda: [0.3333, 0.3333, 0.3333]
    )
    resize_range: tuple[float, float] = (0.4, 1.5)
    gaussian_noise_prob: float = 0
    noise_range: tuple[float, float] = (0, 0)
    poisson_scale_range: tuple[float, float] = (0, 0)
    gray_noise_prob: float = 0
    jpeg_prob: float = 1
    jpeg_range: tuple[float, float] = (75, 95)

    blur_prob2: float = 0
    resize_prob2: list[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])
    resize_mode_list2: list[str] = field(
        default_factory=lambda: ["bilinear", "bicubic", "nearest-exact"]
    )
    resize_mode_prob2: list[float] = field(
        default_factory=lambda: [0.3333, 0.3333, 0.3333]
    )
    resize_range2: tuple[float, float] = (0.6, 1.2)
    gaussian_noise_prob2: float = 0
    noise_range2: tuple[float, float] = (0, 0)
    poisson_scale_range2: tuple[float, float] = (0, 0)
    gray_noise_prob2: float = 0
    jpeg_prob2: float = 1
    jpeg_range2: list[float] = field(default_factory=lambda: [75, 95])

    resize_mode_list3: list[str] = field(
        default_factory=lambda: ["bilinear", "bicubic", "nearest-exact"]
    )
    resize_mode_prob3: list[float] = field(
        default_factory=lambda: [0.3333, 0.3333, 0.3333]
    )

    queue_size: int = 180
    datasets: dict[str, DatasetOptions] = {}
    train: TrainOptions | None = None
    val: ValOptions | None = None
    logger: LogOptions | None = None
    dist_params: dict[str, Any] | None = None
    onnx: OnnxOptions | None = None

    find_unused_parameters: bool = False
