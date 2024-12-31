# ATD
## atd


```
scale: 4
in_chans: 3
img_size: 96
embed_dim: 210
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 16
category_size: 256
num_tokens: 128
reducted_dim: 20
convffn_kernel_size: 5
img_range: 1.0
mlp_ratio: 2
upsampler: pixelshuffle
resi_connection: 1conv
```
## atd_light


```
scale: 4
in_chans: 3
img_size: 64
embed_dim: 48
depths: [6, 6, 6, 6]
num_heads: [6, 6, 6, 6]
window_size: 16
category_size: 128
num_tokens: 64
reducted_dim: 8
convffn_kernel_size: 7
img_range: 1.0
mlp_ratio: 1
upsampler: pixelshuffledirect
resi_connection: 1conv
```
# DAT
## dat


```
scale: 4
in_chans: 3
img_size: 64
img_range: 1.0
split_size: [8, 32]
depth: [6, 6, 6, 6, 6, 6]
embed_dim: 180
num_heads: [6, 6, 6, 6, 6, 6]
expansion_factor: 4
resi_connection: 1conv
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
upsampler: pixelshuffle
```
## dat_s


```
scale: 4
in_chans: 3
img_size: 64
img_range: 1.0
split_size: [8, 16]
depth: [6, 6, 6, 6, 6, 6]
embed_dim: 180
num_heads: [6, 6, 6, 6, 6, 6]
expansion_factor: 2
resi_connection: 1conv
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
upsampler: pixelshuffle
```
## dat_2


```
scale: 4
in_chans: 3
img_size: 64
img_range: 1.0
split_size: [8, 32]
depth: [6, 6, 6, 6, 6, 6]
embed_dim: 180
num_heads: [6, 6, 6, 6, 6, 6]
expansion_factor: 2
resi_connection: 1conv
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
upsampler: pixelshuffle
```
## dat_light


```
scale: 4
in_chans: 3
img_size: 64
img_range: 1.0
split_size: [8, 32]
depth: [18]
embed_dim: 60
num_heads: [6]
expansion_factor: 2
resi_connection: 3conv
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
upsampler: pixelshuffle
```
# DRCT
## drct


```
scale: 4
in_chans: 3
img_size: 64
window_size: 16
img_range: 1.0
depths: [6, 6, 6, 6, 6, 6]
embed_dim: 180
num_heads: [6, 6, 6, 6, 6, 6]
mlp_ratio: 2.0
upsampler: pixelshuffle
resi_connection: 1conv
```
## drct_l


```
scale: 4
in_chans: 3
img_size: 64
window_size: 16
img_range: 1.0
depths: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
embed_dim: 180
num_heads: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
mlp_ratio: 2.0
upsampler: pixelshuffle
resi_connection: 1conv
```
## drct_xl


```
scale: 4
in_chans: 3
img_size: 64
window_size: 16
img_range: 1.0
depths: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
embed_dim: 180
num_heads: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
mlp_ratio: 2.0
upsampler: pixelshuffle
resi_connection: 1conv
```
# HAT
## hat_l


```
scale: 4
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
window_size: 16
compress_ratio: 3
squeeze_factor: 30
conv_scale: 0.01
overlap_ratio: 0.5
mlp_ratio: 2.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: pixelshuffle
resi_connection: 1conv
num_feat: 64
```
## hat_m


```
scale: 4
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 16
compress_ratio: 3
squeeze_factor: 30
conv_scale: 0.01
overlap_ratio: 0.5
mlp_ratio: 2.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: pixelshuffle
resi_connection: 1conv
num_feat: 64
```
## hat_s


```
scale: 4
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 144
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 16
compress_ratio: 24
squeeze_factor: 24
conv_scale: 0.01
overlap_ratio: 0.5
mlp_ratio: 2.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: pixelshuffle
resi_connection: 1conv
num_feat: 64
```
# OmniSR
## omnisr


```
scale: 4
num_in_ch: 3
num_out_ch: 3
num_feat: 64
block_num: 1
pe: true
window_size: 8
res_num: 1
bias: true
```
# PLKSR
## plksr


```
scale: 4
dim: 64
n_blocks: 28
ccm_type: DCCM
kernel_size: 17
split_ratio: 0.25
lk_type: PLK
use_max_kernel: false
sparse_kernels: [5, 5, 5, 5]
sparse_dilations: [1, 2, 3, 4]
with_idt: false
use_ea: true
```
## plksr_tiny


```
scale: 4
dim: 64
n_blocks: 12
ccm_type: DCCM
kernel_size: 13
split_ratio: 0.25
lk_type: PLK
use_max_kernel: false
sparse_kernels: [5, 5, 5, 5]
sparse_dilations: [1, 2, 3, 4]
with_idt: false
use_ea: false
```
# UpCunet4x
## realcugan


```
scale: 4
pro: false
fast: false
in_channels: 3
out_channels: 3
```
# RGT
## rgt


```
scale: 4
img_size: 64
in_chans: 3
embed_dim: 180
depth: [6, 6, 6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6, 6, 6]
mlp_ratio: 2.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
img_range: 1.0
resi_connection: 1conv
split_size: [8, 32]
c_ratio: 0.5
```
## rgt_s


```
scale: 4
img_size: 64
in_chans: 3
embed_dim: 180
depth: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
mlp_ratio: 2.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
img_range: 1.0
resi_connection: 1conv
split_size: [8, 32]
c_ratio: 0.5
```
# RRDBNet
## esrgan


```
scale: 4
use_pixel_unshuffle: true
in_nc: 3
out_nc: 3
num_filters: 64
num_blocks: 23
```
## esrgan_lite


```
scale: 4
use_pixel_unshuffle: true
```
# SPAN
## span


```
num_in_ch: 3
num_out_ch: 3
feature_channels: 48
scale: 4
bias: true
norm: false
img_range: 255.0
rgb_mean: [0.4488, 0.4371, 0.404]
```
# SRFormer
## srformer


```
scale: 4
img_size: 48
patch_size: 1
in_chans: 3
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 24
mlp_ratio: 2.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: pixelshuffle
resi_connection: 1conv
```
## srformer_light


```
scale: 4
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 180
depths: [6, 6, 6, 6]
num_heads: [6, 6, 6, 6]
window_size: 16
mlp_ratio: 2.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: pixelshuffledirect
resi_connection: 1conv
```
# Swin2SR
## swin2sr_l


```
scale: 4
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 240
depths: [6, 6, 6, 6, 6, 6, 6, 6, 6]
num_heads: [8, 8, 8, 8, 8, 8, 8, 8, 8]
window_size: 8
mlp_ratio: 2.0
qkv_bias: true
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: nearest+conv
resi_connection: 3conv
```
## swin2sr_m


```
scale: 4
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 8
mlp_ratio: 2.0
qkv_bias: true
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: pixelshuffle
resi_connection: 1conv
```
## swin2sr_s


```
scale: 4
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 60
depths: [6, 6, 6, 6]
num_heads: [6, 6, 6, 6]
window_size: 8
mlp_ratio: 2.0
qkv_bias: true
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: pixelshuffledirect
resi_connection: 1conv
```
# SwinIR
## swinir_l


```
scale: 4
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 240
depths: [6, 6, 6, 6, 6, 6, 6, 6, 6]
num_heads: [8, 8, 8, 8, 8, 8, 8, 8, 8]
window_size: 8
mlp_ratio: 2.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: nearest+conv
resi_connection: 3conv
start_unshuffle: 1
```
## swinir_m


```
scale: 4
img_size: 48
patch_size: 1
in_chans: 3
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 8
mlp_ratio: 2.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: pixelshuffle
resi_connection: 1conv
start_unshuffle: 1
```
## swinir_s


```
scale: 4
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 60
depths: [6, 6, 6, 6]
num_heads: [6, 6, 6, 6]
window_size: 8
mlp_ratio: 2.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: pixelshuffledirect
resi_connection: 1conv
start_unshuffle: 1
```
# ArtCNN
## artcnn_r16f96


```
in_ch: 3
scale: 4
filters: 96
n_block: 16
kernel_size: 3
```
## artcnn_r8f64


```
in_ch: 3
scale: 4
filters: 64
n_block: 8
kernel_size: 3
```
# EIMN
## eimn_l


```
embed_dims: 64
scale: 2
depths: 1
mlp_ratios: 2.66
drop_rate: 0.0
drop_path_rate: 0.0
num_stages: 16
freeze_param: false
```
## eimn_a


```
embed_dims: 64
scale: 2
depths: 1
mlp_ratios: 2.66
drop_rate: 0.0
drop_path_rate: 0.0
num_stages: 14
freeze_param: false
```
# FlexNet
## FlexNet


```
inp_channels: 3
out_channels: 3
scale: 4
dim: 64
num_blocks: [6, 6, 6, 6, 6, 6]
window_size: 8
hidden_rate: 4
channel_norm: false
attn_drop: 0.0
proj_drop: 0.0
pipeline_type: linear
upsampler: pixelshuffle
```
## metaflexnet


```
inp_channels: 3
out_channels: 3
scale: 4
dim: 64
num_blocks: [4, 6, 6, 8]
window_size: 8
hidden_rate: 4
channel_norm: false
attn_drop: 0.0
proj_drop: 0.0
pipeline_type: meta
upsampler: pixelshuffle
```
# HiT_SIR
## hit_sir


```
scale: 4
in_chans: 3
img_size: 64
base_win_size: [8, 8]
img_range: 1.0
depths: [6, 6, 6, 6]
embed_dim: 60
num_heads: [6, 6, 6, 6]
expansion_factor: 2
resi_connection: 1conv
hier_win_ratios: [0.5, 1, 2, 4, 6, 8]
upsampler: pixelshuffledirect
```
# HiT_SNG
## hit_sng


```
scale: 4
in_chans: 3
img_size: 64
base_win_size: [8, 8]
img_range: 1.0
depths: [6, 6, 6, 6]
embed_dim: 60
num_heads: [6, 6, 6, 6]
expansion_factor: 2
resi_connection: 1conv
hier_win_ratios: [0.5, 1, 2, 4, 6, 8]
upsampler: pixelshuffledirect
```
# HiT_SRF
## hit_srf


```
scale: 4
in_chans: 3
img_size: 64
base_win_size: [8, 8]
img_range: 1.0
depths: [6, 6, 6, 6]
embed_dim: 60
num_heads: [6, 6, 6, 6]
expansion_factor: 2
resi_connection: 1conv
hier_win_ratios: [0.5, 1, 2, 4, 6, 8]
upsampler: pixelshuffledirect
```
# LMLT
## lmlt_base


```
dim: 60
n_blocks: 8
ffn_scale: 2.0
scale: 4
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.0
```
## lmlt_large


```
dim: 84
n_blocks: 8
ffn_scale: 2.0
scale: 4
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.0
```
## lmlt_tiny


```
dim: 36
n_blocks: 8
ffn_scale: 2.0
scale: 4
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.0
```
# MAN
## MAN


```
n_resblocks: 36
n_resgroups: 1
n_colors: 3
n_feats: 180
scale: 2
```
## man_tiny


```
n_resblocks: 5
n_resgroups: 1
n_colors: 3
n_feats: 48
scale: 2
```
## man_light


```
n_resblocks: 24
n_resgroups: 1
n_colors: 3
n_feats: 60
scale: 2
```
# MetaGan2
## MetaGan2


```
in_ch: 3
n_class: 1
dims: [48, 96, 192, 288]
blocks: [3, 3, 9, 3]
downs: [4, 4, 2, 2]
drop_path: 0.2
end_drop: 0.4
```
# MoESR2
## MoESR2


```
in_ch: 3
out_ch: 3
scale: 4
dim: 64
n_blocks: 9
n_block: 4
expansion_factor: 2.6666666666666665
expansion_msg: 1.5
upsampler: pixelshuffledirect
upsample_dim: 64
```
# MoSR
## mosr


```
scale: 4
in_ch: 3
out_ch: 3
n_block: 24
dim: 64
upsampler: pixelshuffle
drop_path: 0.0
kernel_size: 7
expansion_ratio: 1.5
conv_ratio: 1.0
```
## mosr_t


```
scale: 4
in_ch: 3
out_ch: 3
n_block: 5
dim: 48
upsampler: pixelshuffle
drop_path: 0.0
kernel_size: 7
expansion_ratio: 1.5
conv_ratio: 1.0
```
# RCAN
## RCAN


```
scale: 4
n_resgroups: 10
n_resblocks: 20
n_feats: 64
n_colors: 3
rgb_range: 255
norm: true
kernel_size: 3
reduction: 16
res_scale: 1
act_mode: relu
```
## rcan_b


```
scale: 4
n_resgroups: 10
n_resblocks: 20
n_feats: 64
n_colors: 3
rgb_range: 255
norm: false
kernel_size: 3
reduction: 1
res_scale: 1
act_mode: relu
```
## rcan_b2


```
scale: 4
n_resgroups: 6
n_resblocks: 12
n_feats: 64
n_colors: 3
rgb_range: 255
norm: false
kernel_size: 3
reduction: 1
res_scale: 1
act_mode: relu
```
# RealPLKSR
## realplksr


```
in_ch: 3
out_ch: 3
dim: 64
n_blocks: 28
scale: 4
kernel_size: 17
split_ratio: 0.25
use_ea: true
norm_groups: 4
dropout: 0
upsampler: pixelshuffle
layer_norm: true
```
# SCUNet_aaf6aa
## SCUNet_aaf6aa


```
in_nc: 3
out_nc: 3
config: None
dim: 64
drop_path_rate: 0.0
input_resolution: 256
scale: 1
residual: true
state: None
```
# SpanPlus
## spanplus


```
scale: 4
num_in_ch: 3
num_out_ch: 3
blocks: None
feature_channels: 48
drop_rate: 0.0
upsampler: dys
```
## spanplus_sts


```
scale: 4
num_in_ch: 3
num_out_ch: 3
blocks: [2]
feature_channels: 32
drop_rate: 0.0
upsampler: ps
```
## spanplus_s


```
scale: 4
num_in_ch: 3
num_out_ch: 3
blocks: [2]
feature_channels: 32
drop_rate: 0.0
upsampler: dys
```
## spanplus_st


```
scale: 4
num_in_ch: 3
num_out_ch: 3
blocks: None
feature_channels: 48
drop_rate: 0.0
upsampler: ps
```
# SRVGGNetCompact
## compact


```
num_in_ch: 3
num_out_ch: 3
num_feat: 64
num_conv: 16
scale: 4
act_type: prelu
learn_residual: true
```
## ultracompact


```
num_in_ch: 3
num_out_ch: 3
num_feat: 64
num_conv: 8
scale: 4
act_type: prelu
learn_residual: true
```
## superultracompact


```
num_in_ch: 3
num_out_ch: 3
num_feat: 24
num_conv: 8
scale: 4
act_type: prelu
learn_residual: true
```
# TSCUNet
## TSCUNet


```
in_nc: 3
out_nc: 3
clip_size: 5
nb: 2
dim: 64
drop_path_rate: 0.0
input_resolution: 256
scale: 2
residual: true
sigma: false
state: None
```
