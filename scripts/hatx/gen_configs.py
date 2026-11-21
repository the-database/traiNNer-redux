V = {
    "A": ("hat_l_aligned", {}, "experiments/pretrained_models/4x_Diverseg_IP_Redux_HAT_L_aligned_bf16_850k.safetensors"),
    "B": ("hatx_l", {"big_blocks": [2, 5]}, "experiments/hatx_init/4x_hatx_B.safetensors"),
    "C1": ("hatx_l", {"channel": True}, "experiments/hatx_init/4x_hatx_C1.safetensors"),
    "E": ("hatx_l", {"big_blocks": [2, 5], "channel": True, "use_cab": False, "cab_fade": 10000}, "experiments/hatx_init/4x_hatx_E.safetensors"),
    "C1g": ("hatx_l", {"channel": True, "channel_gain": 100.0}, "experiments/hatx_init/4x_hatx_C1.safetensors"),
    "C2": ("hatx_l", {"channel": True, "channel_gain": 100.0, "use_cab": False, "cab_fade": 20000}, "experiments/hatx_init/4x_hatx_C2.safetensors"),
}
# protocol per variant: batch, lr, milestones, total iterations, validation interval. P1: the first
# round (batch 8, 2e-5: loses ~0.1 dB on the 850k start). P2: the 850k run's batch (8 x 4 GPUs) and an
# LR near its final 6.25e-6.
P1 = dict(bs=8, lr="2e-5", m1=20000, m2=30000, total=40000, vf=2500, wu=-1)
P2 = dict(bs=32, lr="1e-5", m1=10000, m2=15000, total=20000, vf=1250, wu=-1)
# P3: P2 at the 850k run's final LR, held constant, after a 1k warmup (P2 still loses ~0.07 dB)
P3 = dict(bs=32, lr="6.25e-6", m1=1000000, m2=2000000, total=20000, vf=1250, wu=1000)
# (a 96 LR patch doesn't fit DiverSeg: 88% of its LR crops are under 96 on a side; the smallest is 64)
P = {"A2": P2, "B2": P2, "C2b": P2, "A3": P3, "Bq": P2, "Ball": P2}
V["A2"] = V["A"]
V["B2"] = V["B"]
V["C2b"] = (V["C2"][0], {**V["C2"][1], "cab_fade": 10000}, V["C2"][2])
V["A3"] = V["A"]
V["Bq"] = ("hatx_l", {"big_blocks": [1, 2, 4, 5]}, "experiments/hatx_init/4x_hatx_Bq.safetensors")
V["Ball"] = ("hatx_l", {"big_blocks": [0, 1, 2, 3, 4, 5]}, "experiments/hatx_init/4x_hatx_Ball.safetensors")
N = {"A2": "A", "B2": "B", "C2b": "C2", "A3": "A_lr6e-6c"}  # run names (batch is in the suffix)
T = """name: 4x_hatx_{v}_ft850k_b{bsn}lq{lq}
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
    lq_size: {lq}
    use_hflip: true
    use_rot: true
    num_worker_per_gpu: 10
    batch_size_per_gpu: {bs}
    accum_iter: {ac}
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
    lr: !!float {lr}
    weight_decay: 0
    betas: [0.9, 0.99]
  scheduler:
    type: MultiStepLR
    milestones: [{m1}, {m2}]
    gamma: 0.5
  total_iter: {total}
  warmup_iter: {wu}
  losses:
    - type: charbonnierloss
      loss_weight: 1.0

val:
  val_enabled: true
  val_freq: {vf}
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
    pr = {"lq": 64, "ac": 1, **P.get(v, P1)}
    pr["bsn"] = f"{pr['bs']}x{pr['ac']}" if pr["ac"] > 1 else pr["bs"]
    name_v = N.get(v, v)
    open(f"options/train/hatx/4x_hatx_{v}.yml", "w").write(T.format(v=name_v, arch=arch, extra=extra, init=init, **pr))
