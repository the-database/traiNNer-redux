V = {
    "A": ("hat_l_aligned", {}, "experiments/pretrained_models/4x_Diverseg_IP_Redux_HAT_L_aligned_bf16_850k.safetensors"),
    "B": ("hatx_l", {"big_blocks": [2, 5]}, "experiments/hatx_init/4x_hatx_B.safetensors"),
    "C1": ("hatx_l", {"channel": True}, "experiments/hatx_init/4x_hatx_C1.safetensors"),
    "E": ("hatx_l", {"big_blocks": [2, 5], "channel": True, "use_cab": False, "cab_fade": 10000}, "experiments/hatx_init/4x_hatx_E.safetensors"),
    "C1g": ("hatx_l", {"channel": True, "channel_gain": 100.0}, "experiments/hatx_init/4x_hatx_C1.safetensors"),
}
T = """name: 4x_hatx_{v}_ft850k_b8lq64
scale: 4
use_amp: true
amp_bf16: true
use_channels_last: true
fast_matmul: false
num_gpu: auto
use_compile: false
manual_seed: 1024

datasets:
  train:
    name: DiverSeg-IP
    type: pairedimagedataset
    dataroot_gt: ../datasets/DiverSeg/HR_cropped iqa 0.5
    dataroot_lq: ../datasets/DiverSeg/LR_bicubic iqa 0.5/x4
    lq_size: 64
    use_hflip: true
    use_rot: true
    num_worker_per_gpu: 10
    batch_size_per_gpu: 8
    accum_iter: 1
  val:
    name: Urban100
    type: pairedimagedataset
    dataroot_gt: datasets/benchmark/Urban100/HR
    dataroot_lq: datasets/benchmark/Urban100/LR_bicubic/X4
    filename_tmpl: "{{}}x4"

network_g:
  type: {arch}
  use_checkpoint: true
{extra}
path:
  pretrain_network_g: {init}
  strict_load_g: true
  resume_state: ~

train:
  ema_decay: 0.999
  ema_power: 0.75
  grad_clip: false
  optim_g:
    type: AdamW
    lr: !!float 2e-5
    weight_decay: 0
    betas: [0.9, 0.99]
  scheduler:
    type: MultiStepLR
    milestones: [20000, 30000]
    gamma: 0.5
  total_iter: 40000
  warmup_iter: -1
  losses:
    - type: charbonnierloss
      loss_weight: 1.0

val:
  val_enabled: true
  val_freq: 2500
  save_img: false
  metrics_enabled: true
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: true
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: true

logger:
  print_freq: 100
  save_checkpoint_freq: 10000
  save_checkpoint_format: safetensors
  use_tb_logger: true
"""
import json
for v, (arch, kw, init) in V.items():
    extra = "".join(f"  {k}: {json.dumps(x)}\n" for k, x in kw.items())
    open(f"options/train/hatx/4x_hatx_{v}.yml", "w").write(T.format(v=v, arch=arch, extra=extra, init=init))
