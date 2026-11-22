# Denoise fine-tunes on rc40 (the IllustrationJaNai denoise set), with the DAT-2 denoise release's
# recipe: losses, GAN (wgan_softplus, dunet), lr 1e-4, batch 32 (8 x 4 GPUs there, 32 x 1 here), LR
# patch 64. Each starts from its own 20k DiverSeg fine-tune (all from HAT-L aligned 850k), and ~100
# rc40 sources are held out (scripts/hatx/split_pairs.py) for validation.
# Run from the repo root: python scripts/hatx/gen_dn_configs.py
import json

V = {
    "A": ("hat_l_aligned", {}, "experiments/4x_hatx_A_ft850k_b32lq64/models/net_g_ema_20000.safetensors"),
    "Bq": ("hatx_l", {"big_blocks": [1, 2, 4, 5]}, "experiments/4x_hatx_Bq_ft850k_b32lq64/models/net_g_ema_20000.safetensors"),
    "Ball": ("hatx_l", {"big_blocks": [0, 1, 2, 3, 4, 5]}, "experiments/4x_hatx_Ball_ft850k_b32lq64/models/net_g_ema_20000.safetensors"),
    "Ballc": (
        "hatx_l",
        {"big_blocks": [0, 1, 2, 3, 4, 5], "channel": True, "channel_gain": 100.0},
        "experiments/hatx_init/4x_hatx_Ballc_from_Ball20k.safetensors",
    ),
}
T = """name: 4x_hatx_dn_{v}_rc40ho_b{bs}lq64
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
    name: rc40 (held-out sources removed)
    type: pairedimagedataset
    dataroot_gt: datasets/rc40_train/hr
    dataroot_lq: datasets/rc40_train/lr
    lq_size: 64
    use_hflip: true
    use_rot: true
    num_worker_per_gpu: 10
    batch_size_per_gpu: {bs}
    accum_iter: 1
  val:
    name: rc40 held out
    type: pairedimagedataset
    dataroot_gt: datasets/rc40_val100/hr
    dataroot_lq: datasets/rc40_val100/lr

network_g:
  type: {arch}
  use_checkpoint: true
{extra}
network_d:
  type: dunet

path:
  pretrain_network_g: {init}
  strict_load_g: true
  resume_state: ~

train:
  ema_decay: 0.999
  grad_clip: false
  optim_g:
    type: AdamW
    lr: !!float 1e-4
    weight_decay: 0
    betas: [0.9, 0.99]
  optim_d:
    type: AdamW
    lr: !!float 1e-4
    weight_decay: 0
    betas: [0.9, 0.99]
  scheduler:
    type: MultiStepLR
    milestones: [100000]
    gamma: 0.5
  total_iter: 30000
  warmup_iter: -1
  losses:
    - type: mssimloss
      include_luminance: false
      loss_weight: 0.5
    - type: cosimloss
      loss_weight: 1.0
    - type: perceptualloss
      criterion: charbonnier
      loss_weight: 0.01
    - type: hsluvloss
      downscale_factor: 1
      criterion: charbonnier
      loss_weight: 1.0
    - type: ganloss
      gan_type: wgan_softplus
      loss_weight: 0.1

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
  save_checkpoint_freq: 2500
  save_checkpoint_format: safetensors
  use_tb_logger: true
"""
for v, (arch, kw, init) in V.items():
    extra = "".join(f"  {k}: {json.dumps(x)}\n" for k, x in kw.items())
    open(f"options/train/hatx/4x_hatx_dn_{v}.yml", "w").write(T.format(v=v, arch=arch, extra=extra, init=init, bs=32))
