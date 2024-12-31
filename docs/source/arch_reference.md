# atd, atd_light


```
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 90
depths: [6, 6, 6, 6]
num_heads: [6, 6, 6, 6]
window_size: 8
category_size: 256
num_tokens: 64
reducted_dim: 4
convffn_kernel_size: 5
mlp_ratio: 2.0
qkv_bias: true
ape: false
patch_norm: true
upscale: 1
img_range: 1.0
upsampler: 
resi_connection: 1conv
norm: true
```
# dat, dat_s, dat_2, dat_light


```
img_size: 64
in_chans: 3
embed_dim: 180
split_size: [2, 4]
depth: [2, 2, 2, 2]
num_heads: [2, 2, 2, 2]
expansion_factor: 4.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
upscale: 2
img_range: 1.0
resi_connection: 1conv
upsampler: pixelshuffle
```
# drct, drct_l, drct_xl


```
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 16
mlp_ratio: 2.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
upscale: 1
img_range: 1.0
upsampler: 
resi_connection: 1conv
gc: 32
```
# hat_l, hat_m, hat_s


```
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 96
depths: [6, 6, 6, 6]
num_heads: [6, 6, 6, 6]
window_size: 7
compress_ratio: 3
squeeze_factor: 30
conv_scale: 0.01
overlap_ratio: 0.5
mlp_ratio: 4.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
upscale: 1
img_range: 1.0
upsampler: pixelshuffle
resi_connection: 1conv
num_feat: 64
```
# omnisr


```
num_in_ch: 3
num_out_ch: 3
num_feat: 64
block_num: 1
pe: true
window_size: 8
res_num: 1
up_scale: 4
bias: true
```
# plksr, plksr_tiny


```
dim: 64
n_blocks: 28
upscaling_factor: 4
ccm_type: CCM
kernel_size: 17
split_ratio: 0.25
lk_type: PLK
use_max_kernel: false
sparse_kernels: [5, 5, 5, 5]
sparse_dilations: [1, 2, 3, 4]
with_idt: false
use_ea: true
```
# realcugan


```
in_channels: 3
out_channels: 3
pro: false
```
# rgt, rgt_s


```
img_size: 64
in_chans: 3
embed_dim: 180
depth: [2, 2, 2, 2]
num_heads: [2, 2, 2, 2]
mlp_ratio: 4.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
upscale: 2
img_range: 1.0
resi_connection: 1conv
split_size: [8, 8]
c_ratio: 0.5
```
# esrgan, esrgan_lite


```
in_nc: 3
out_nc: 3
num_filters: 64
num_blocks: 23
scale: 4
plus: false
shuffle_factor: None
norm: None
act: leakyrelu
upsampler: upconv
mode: CNA
```
# span


```
feature_channels: 48
upscale: 4
bias: true
norm: true
img_range: 255.0
rgb_mean: [0.4488, 0.4371, 0.404]
```
# srformer, srformer_light


```
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 96
depths: [6, 6, 6, 6]
num_heads: [6, 6, 6, 6]
window_size: 7
mlp_ratio: 4.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
upscale: 1
img_range: 1.0
upsampler: 
resi_connection: 1conv
```
# swin2sr_l, swin2sr_m, swin2sr_s


```
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 96
depths: [6, 6, 6, 6]
num_heads: [6, 6, 6, 6]
window_size: 7
mlp_ratio: 4.0
qkv_bias: true
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
upscale: 2
img_range: 1.0
upsampler: 
resi_connection: 1conv
```
# swinir_l, swinir_m, swinir_s


```
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 96
depths: [6, 6, 6, 6]
num_heads: [6, 6, 6, 6]
window_size: 7
mlp_ratio: 4.0
qkv_bias: true
qk_scale: None
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
upscale: 1
img_range: 1.0
upsampler: 
resi_connection: 1conv
start_unshuffle: 1
```
# artcnn_r16f96, artcnn_r8f64


```
in_ch: 3
scale: 4
filters: 96
n_block: 16
kernel_size: 3
```
# eimn_l, eimn_a


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
# flexnet, metaflexnet


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
# han


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
# hit_sir


```
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 60
depths: None
num_heads: None
base_win_size: None
mlp_ratio: 2.0
drop_rate: 0.0
value_drop_rate: 0.0
drop_path_rate: 0.0
ape: false
patch_norm: true
use_checkpoint: false
upscale: 4
img_range: 1.0
upsampler: pixelshuffledirect
resi_connection: 1conv
hier_win_ratios: None
```
# hit_sng


```
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 60
depths: None
num_heads: None
base_win_size: None
mlp_ratio: 2.0
drop_rate: 0.0
value_drop_rate: 0.0
drop_path_rate: 0.0
ape: false
patch_norm: true
use_checkpoint: false
upscale: 4
img_range: 1.0
upsampler: pixelshuffledirect
resi_connection: 1conv
hier_win_ratios: None
```
# hit_srf


```
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 60
depths: None
num_heads: None
base_win_size: None
mlp_ratio: 2.0
drop_rate: 0.0
value_drop_rate: 0.0
drop_path_rate: 0.0
ape: false
patch_norm: true
use_checkpoint: false
upscale: 4
img_range: 1.0
upsampler: pixelshuffledirect
resi_connection: 1conv
hier_win_ratios: None
```
# lmlt_base, lmlt_large, lmlt_tiny


```
n_blocks: 8
ffn_scale: 2.0
scale: 4
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.0
```
# man, man_tiny, man_light


```
n_resblocks: 36
n_resgroups: 1
n_colors: 3
n_feats: 180
scale: 2
res_scale: 1.0
```
# metagan2


```
in_ch: 3
n_class: 1
dims: [48, 96, 192, 288]
blocks: [3, 3, 9, 3]
downs: [4, 4, 2, 2]
drop_path: 0.2
end_drop: 0.4
```
# moesr2


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
# mosr, mosr_t


```
in_ch: 3
out_ch: 3
upscale: 4
n_block: 24
dim: 64
upsampler: ps
drop_path: 0.0
kernel_size: 7
expansion_ratio: 1.5
conv_ratio: 1.0
```
# rcan, rcan_b, rcan_b2


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
# realplksr


```
in_ch: 3
out_ch: 3
dim: 64
n_blocks: 28
upscaling_factor: 4
kernel_size: 17
split_ratio: 0.25
use_ea: true
norm_groups: 4
dropout: 0
upsampler: pixelshuffle
layer_norm: true
```
# scunet_aaf6aa


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
# spanplus, spanplus_sts, spanplus_s, spanplus_st


```
num_in_ch: 3
num_out_ch: 3
blocks: None
feature_channels: 48
upscale: 4
drop_rate: 0.0
upsampler: dys
```
# compact, ultracompact, superultracompact


```
num_in_ch: 3
num_out_ch: 3
num_feat: 64
num_conv: 16
upscale: 4
act_type: prelu
learn_residual: true
```
# tscunet


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
