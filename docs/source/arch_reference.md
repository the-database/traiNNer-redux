# Architecture reference
This page lists all available parameters for each architecture in traiNNer-redux. While the default configs use the official defaults and shouldn't need to be modified by most users, advanced users may wish to inspect or modify architectures to train to their liking. Please keep in mind that changing parameters for generator architectures can affect compatibility with using pretrain models.
## Generator architectures (`network_g`)
### ATD
#### atd


```yaml
type: atd
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
#### atd_light


```yaml
type: atd_light
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
### ArtCNN
#### artcnn_r16f96


```yaml
type: artcnn_r16f96
in_ch: 3
filters: 96
n_block: 16
kernel_size: 3
```
#### artcnn_r8f48


```yaml
type: artcnn_r8f48
in_ch: 3
filters: 48
n_block: 8
kernel_size: 3
```
#### artcnn_r8f64


```yaml
type: artcnn_r8f64
in_ch: 3
filters: 64
n_block: 8
kernel_size: 3
```
### CRAFT
#### craft


```yaml
type: craft
window_size: 16
embed_dim: 48
depths: [2, 2, 2, 2]
num_heads: [6, 6, 6, 6]
split_size_0: 4
split_size_1: 16
mlp_ratio: 2.0
qkv_bias: true
qk_scale: ~
img_range: 1.0
resi_connection: 1conv
```
### CascadedGaze
#### cascadedgaze


```yaml
type: cascadedgaze
img_channel: 3
width: 60
middle_blk_num: 10
enc_blk_nums: ~
dec_blk_nums: ~
GCE_CONVS_nums: ~
```
### DAT
#### dat


```yaml
type: dat
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
qk_scale: ~
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
upsampler: pixelshuffle
```
#### dat_2


```yaml
type: dat_2
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
qk_scale: ~
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
upsampler: pixelshuffle
```
#### dat_light


```yaml
type: dat_light
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
qk_scale: ~
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
upsampler: pixelshuffle
```
#### dat_s


```yaml
type: dat_s
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
qk_scale: ~
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
upsampler: pixelshuffle
```
### DCTLSA
#### dctlsa


```yaml
type: dctlsa
in_nc: 3
nf: 55
num_modules: 6
out_nc: 3
num_head: 5
```
### DITN_Real
#### ditn_real


```yaml
type: ditn_real
inp_channels: 3
dim: 60
ITL_blocks: 4
SAL_blocks: 4
UFONE_blocks: 1
ffn_expansion_factor: 2
bias: false
LayerNorm_type: WithBias
patch_size: 8
```
### DRCT
#### drct


```yaml
type: drct
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
#### drct_l


```yaml
type: drct_l
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
#### drct_xl


```yaml
type: drct_xl
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
### DWT
#### dwt


```yaml
type: dwt
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 16
mlp_ratio: 2.0
qkv_bias: true
qk_scale: ~
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: pixelshuffle
resi_connection: 1conv
layer_kinds: [[0, -1, 0, -1, 0, -1], [0, -1, 0, -1, 0, -1], [0, 2, 0, 2, 0, 2], [0, 2, 0, 2, 0, 2], [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]]
```
#### dwt_s


```yaml
type: dwt_s
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 8
mlp_ratio: 2.0
qkv_bias: true
qk_scale: ~
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
ape: false
patch_norm: true
use_checkpoint: false
img_range: 1.0
upsampler: pixelshuffle
resi_connection: 1conv
layer_kinds: [[0, -1, 0, -1, 0, -1], [0, -1, 0, -1, 0, -1], [0, 2, 0, 2, 0, 2], [0, 2, 0, 2, 0, 2], [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]]
```
### EIMN
#### eimn_a


```yaml
type: eimn_a
embed_dims: 64
depths: 1
mlp_ratios: 2.66
drop_rate: 0.0
drop_path_rate: 0.0
num_stages: 14
freeze_param: false
```
#### eimn_l


```yaml
type: eimn_l
embed_dims: 64
depths: 1
mlp_ratios: 2.66
drop_rate: 0.0
drop_path_rate: 0.0
num_stages: 16
freeze_param: false
```
### ELAN
#### elan


```yaml
type: elan
colors: 3
rgb_range: 255
norm: false
window_sizes: ~
m_elan: 36
c_elan: 180
n_share: 0
r_expand: 2
```
#### elan_light


```yaml
type: elan_light
colors: 3
rgb_range: 255
norm: false
window_sizes: ~
m_elan: 24
c_elan: 60
n_share: 1
r_expand: 2
```
### EMT
#### emt


```yaml
type: emt
num_in_ch: 3
num_out_ch: 3
upsampler: pixelshuffle
dim: 60
n_blocks: 6
n_layers: 6
num_heads: 3
mlp_ratio: 2
n_GTLs: 2
window_list: [[32, 8], [8, 32]]
shift_list: [[16, 4], [4, 16]]
```
### FlexNet
#### flexnet


```yaml
type: flexnet
inp_channels: 3
out_channels: 3
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
#### metaflexnet


```yaml
type: metaflexnet
inp_channels: 3
out_channels: 3
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
### GRL
#### grl_b


```yaml
type: grl_b
img_size: 64
in_channels: 3
out_channels: 3
embed_dim: 180
img_range: 1.0
upsampler: pixelshuffle
depths: [4, 4, 8, 8, 8, 4, 4]
num_heads_window: [3, 3, 3, 3, 3, 3, 3]
num_heads_stripe: [3, 3, 3, 3, 3, 3, 3]
window_size: 32
stripe_size: [64, 64]
stripe_groups: [None, None]
stripe_shift: true
mlp_ratio: 2.0
qkv_bias: true
qkv_proj_type: linear
anchor_proj_type: avgpool
anchor_one_stage: true
anchor_window_down_factor: 4
out_proj_type: linear
local_connection: true
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
pretrained_window_size: [0, 0]
pretrained_stripe_size: [0, 0]
conv_type: 1conv
init_method: n
fairscale_checkpoint: false
offload_to_cpu: false
euclidean_dist: false
```
#### grl_s


```yaml
type: grl_s
img_size: 64
in_channels: 3
out_channels: 3
embed_dim: 128
img_range: 1.0
upsampler: pixelshuffle
depths: [4, 4, 4, 4]
num_heads_window: [2, 2, 2, 2]
num_heads_stripe: [2, 2, 2, 2]
window_size: 32
stripe_size: [64, 64]
stripe_groups: [None, None]
stripe_shift: true
mlp_ratio: 2.0
qkv_bias: true
qkv_proj_type: linear
anchor_proj_type: avgpool
anchor_one_stage: true
anchor_window_down_factor: 4
out_proj_type: linear
local_connection: false
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
pretrained_window_size: [0, 0]
pretrained_stripe_size: [0, 0]
conv_type: 1conv
init_method: n
fairscale_checkpoint: false
offload_to_cpu: false
euclidean_dist: false
```
#### grl_t


```yaml
type: grl_t
img_size: 64
in_channels: 3
out_channels: 3
embed_dim: 64
img_range: 1.0
upsampler: pixelshuffledirect
depths: [4, 4, 4, 4]
num_heads_window: [2, 2, 2, 2]
num_heads_stripe: [2, 2, 2, 2]
window_size: 32
stripe_size: [64, 64]
stripe_groups: [None, None]
stripe_shift: true
mlp_ratio: 2.0
qkv_bias: true
qkv_proj_type: linear
anchor_proj_type: avgpool
anchor_one_stage: true
anchor_window_down_factor: 4
out_proj_type: linear
local_connection: false
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
pretrained_window_size: [0, 0]
pretrained_stripe_size: [0, 0]
conv_type: 1conv
init_method: n
fairscale_checkpoint: false
offload_to_cpu: false
euclidean_dist: false
```
### HAT
#### hat_l


```yaml
type: hat_l
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
qk_scale: ~
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
#### hat_m


```yaml
type: hat_m
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
qk_scale: ~
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
#### hat_s


```yaml
type: hat_s
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
qk_scale: ~
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
### HiT_SIR
#### hit_sir


```yaml
type: hit_sir
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
### HiT_SNG
#### hit_sng


```yaml
type: hit_sng
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
### HiT_SRF
#### hit_srf


```yaml
type: hit_srf
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
### LMLT
#### lmlt_base


```yaml
type: lmlt_base
dim: 60
n_blocks: 8
ffn_scale: 2.0
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.0
```
#### lmlt_large


```yaml
type: lmlt_large
dim: 84
n_blocks: 8
ffn_scale: 2.0
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.0
```
#### lmlt_tiny


```yaml
type: lmlt_tiny
dim: 36
n_blocks: 8
ffn_scale: 2.0
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.0
```
### MAN
#### man


```yaml
type: man
n_resblocks: 36
n_resgroups: 1
n_colors: 3
n_feats: 180
```
#### man_light


```yaml
type: man_light
n_resblocks: 24
n_resgroups: 1
n_colors: 3
n_feats: 60
```
#### man_tiny


```yaml
type: man_tiny
n_resblocks: 5
n_resgroups: 1
n_colors: 3
n_feats: 48
```
### MoESR2
#### moesr2


```yaml
type: moesr2
in_ch: 3
out_ch: 3
dim: 64
n_blocks: 9
n_block: 4
expansion_factor: 2.6666666666666665
expansion_msg: 1.5
upsampler: pixelshuffledirect
upsample_dim: 64
```
### MoSR
#### mosr


```yaml
type: mosr
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
#### mosr_t


```yaml
type: mosr_t
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
### MoSRv2
#### mosrv2


```yaml
type: mosrv2
in_ch: 3
n_block: 24
dim: 64
upsampler: pixelshuffledirect
expansion_ratio: 1.5
mid_dim: 32
unshuffle_mod: true
```
### OmniSR
#### omnisr


```yaml
type: omnisr
num_in_ch: 3
num_out_ch: 3
num_feat: 64
block_num: 1
pe: true
window_size: 8
res_num: 5
bias: true
```
### PLKSR
#### plksr


```yaml
type: plksr
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
#### plksr_tiny


```yaml
type: plksr_tiny
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
### RCAN
#### rcan


```yaml
type: rcan
n_resgroups: 10
n_resblocks: 20
n_feats: 64
n_colors: 3
rgb_range: 255
norm: false
kernel_size: 3
reduction: 16
res_scale: 1
act_mode: relu
unshuffle_mod: false
```
#### rcan_l


```yaml
type: rcan_l
n_resgroups: 10
n_resblocks: 20
n_feats: 96
n_colors: 3
rgb_range: 255
norm: false
kernel_size: 3
reduction: 16
res_scale: 1
act_mode: relu
unshuffle_mod: false
```
#### rcan_unshuffle


```yaml
type: rcan_unshuffle
n_resgroups: 10
n_resblocks: 20
n_feats: 64
n_colors: 3
rgb_range: 255
norm: false
kernel_size: 3
reduction: 16
res_scale: 1
act_mode: relu
unshuffle_mod: true
```
### RGT
#### rgt


```yaml
type: rgt
img_size: 64
in_chans: 3
embed_dim: 180
depth: [6, 6, 6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6, 6, 6]
mlp_ratio: 2.0
qkv_bias: true
qk_scale: ~
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
img_range: 1.0
resi_connection: 1conv
split_size: [8, 32]
c_ratio: 0.5
```
#### rgt_s


```yaml
type: rgt_s
img_size: 64
in_chans: 3
embed_dim: 180
depth: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
mlp_ratio: 2.0
qkv_bias: true
qk_scale: ~
drop_rate: 0.0
attn_drop_rate: 0.0
drop_path_rate: 0.1
use_chk: false
img_range: 1.0
resi_connection: 1conv
split_size: [8, 32]
c_ratio: 0.5
```
### RRDBNet
#### esrgan


```yaml
type: esrgan
use_pixel_unshuffle: true
in_nc: 3
out_nc: 3
num_filters: 64
num_blocks: 23
```
#### esrgan_lite


```yaml
type: esrgan_lite
use_pixel_unshuffle: true
in_nc: 3
out_nc: 3
num_filters: 32
num_blocks: 12
```
### RTMoSR
#### rtmosr


```yaml
type: rtmosr
dim: 32
ffn_expansion: 2
n_blocks: 2
unshuffle_mod: false
dccm: true
se: true
```
#### rtmosr_l


```yaml
type: rtmosr_l
dim: 32
ffn_expansion: 2
n_blocks: 2
unshuffle_mod: true
dccm: true
se: true
```
#### rtmosr_ul


```yaml
type: rtmosr_ul
dim: 32
ffn_expansion: 1.5
n_blocks: 2
unshuffle_mod: true
dccm: false
se: true
```
### RealPLKSR
#### realplksr


```yaml
type: realplksr
in_ch: 3
out_ch: 3
dim: 64
n_blocks: 28
kernel_size: 17
split_ratio: 0.25
use_ea: true
norm_groups: 4
dropout: 0
upsampler: pixelshuffle
layer_norm: true
```
#### realplksr_large


```yaml
type: realplksr_large
in_ch: 3
out_ch: 3
dim: 96
n_blocks: 28
kernel_size: 17
split_ratio: 0.25
use_ea: true
norm_groups: 4
dropout: 0
upsampler: pixelshuffle
layer_norm: true
```
#### realplksr_tiny


```yaml
type: realplksr_tiny
in_ch: 3
out_ch: 3
dim: 64
n_blocks: 12
kernel_size: 13
split_ratio: 0.25
use_ea: false
norm_groups: 4
dropout: 0
upsampler: pixelshuffle
layer_norm: true
```
### SAFMN
#### safmn


```yaml
type: safmn
dim: 36
n_blocks: 8
ffn_scale: 2.0
```
#### safmn_l


```yaml
type: safmn_l
dim: 128
n_blocks: 16
ffn_scale: 2.0
```
### SCUNet_aaf6aa
#### scunet_aaf6aa


```yaml
type: scunet_aaf6aa
in_nc: 3
out_nc: 3
config: ~
dim: 64
drop_path_rate: 0.0
input_resolution: 256
residual: true
state: ~
```
### SPAN
#### span


```yaml
type: span
num_in_ch: 3
num_out_ch: 3
feature_channels: 52
bias: true
norm: false
img_range: 255.0
rgb_mean: [0.4488, 0.4371, 0.404]
```
#### span_s


```yaml
type: span_s
num_in_ch: 3
num_out_ch: 3
feature_channels: 48
bias: true
norm: false
img_range: 255.0
rgb_mean: [0.4488, 0.4371, 0.404]
```
### SRFormer
#### srformer


```yaml
type: srformer
img_size: 48
patch_size: 1
in_chans: 3
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 24
mlp_ratio: 2.0
qkv_bias: true
qk_scale: ~
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
#### srformer_light


```yaml
type: srformer_light
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 180
depths: [6, 6, 6, 6]
num_heads: [6, 6, 6, 6]
window_size: 16
mlp_ratio: 2.0
qkv_bias: true
qk_scale: ~
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
### SRVGGNetCompact
#### compact


```yaml
type: compact
num_in_ch: 3
num_out_ch: 3
num_feat: 64
num_conv: 16
act_type: prelu
learn_residual: true
```
#### superultracompact


```yaml
type: superultracompact
num_in_ch: 3
num_out_ch: 3
num_feat: 24
num_conv: 8
act_type: prelu
learn_residual: true
```
#### ultracompact


```yaml
type: ultracompact
num_in_ch: 3
num_out_ch: 3
num_feat: 64
num_conv: 8
act_type: prelu
learn_residual: true
```
### Sebica
#### sebica


```yaml
type: sebica
N: 16
```
#### sebica_mini


```yaml
type: sebica_mini
N: 8
```
### SeemoRe
#### seemore_t


```yaml
type: seemore_t
in_chans: 3
num_experts: 3
num_layers: 6
embedding_dim: 36
img_range: 1.0
use_shuffle: true
global_kernel_size: 11
recursive: 2
lr_space: exp
topk: 1
```
### SpanPlus
#### spanplus


```yaml
type: spanplus
num_in_ch: 3
num_out_ch: 3
blocks: ~
feature_channels: 48
drop_rate: 0.0
upsampler: dys
```
#### spanplus_s


```yaml
type: spanplus_s
num_in_ch: 3
num_out_ch: 3
blocks: [2]
feature_channels: 32
drop_rate: 0.0
upsampler: dys
```
#### spanplus_st


```yaml
type: spanplus_st
num_in_ch: 3
num_out_ch: 3
blocks: ~
feature_channels: 48
drop_rate: 0.0
upsampler: ps
```
#### spanplus_sts


```yaml
type: spanplus_sts
num_in_ch: 3
num_out_ch: 3
blocks: [2]
feature_channels: 32
drop_rate: 0.0
upsampler: ps
```
### Swin2SR
#### swin2sr_l


```yaml
type: swin2sr_l
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
#### swin2sr_m


```yaml
type: swin2sr_m
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
#### swin2sr_s


```yaml
type: swin2sr_s
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
### SwinIR
#### swinir_l


```yaml
type: swinir_l
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 240
depths: [6, 6, 6, 6, 6, 6, 6, 6, 6]
num_heads: [8, 8, 8, 8, 8, 8, 8, 8, 8]
window_size: 8
mlp_ratio: 2.0
qkv_bias: true
qk_scale: ~
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
#### swinir_m


```yaml
type: swinir_m
img_size: 48
patch_size: 1
in_chans: 3
embed_dim: 180
depths: [6, 6, 6, 6, 6, 6]
num_heads: [6, 6, 6, 6, 6, 6]
window_size: 8
mlp_ratio: 2.0
qkv_bias: true
qk_scale: ~
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
#### swinir_s


```yaml
type: swinir_s
img_size: 64
patch_size: 1
in_chans: 3
embed_dim: 60
depths: [6, 6, 6, 6]
num_heads: [6, 6, 6, 6]
window_size: 8
mlp_ratio: 2.0
qkv_bias: true
qk_scale: ~
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
### TSCUNet
#### tscunet


```yaml
type: tscunet
in_nc: 3
out_nc: 3
clip_size: 5
nb: 2
dim: 64
drop_path_rate: 0.0
input_resolution: 256
residual: true
sigma: false
state: ~
```
### UpCunet4x
#### realcugan


```yaml
type: realcugan
pro: false
fast: false
in_channels: 3
out_channels: 3
```
## Discriminator architectures (`network_d`)
### DUnet
#### dunet


```yaml
type: dunet
num_in_ch: 3
num_feat: 64
```
### MetaGan2
#### metagan2


```yaml
type: metagan2
in_ch: 3
n_class: 1
dims: [48, 96, 192, 288]
blocks: [3, 3, 9, 3]
downs: [4, 4, 2, 2]
drop_path: 0.0
end_drop: 0.0
```
### UNetDiscriminatorSN
#### unetdiscriminatorsn


```yaml
type: unetdiscriminatorsn
num_in_ch: 3
num_feat: 64
skip_connection: true
```
### VGGStyleDiscriminator
#### vggstylediscriminator


```yaml
type: vggstylediscriminator
num_in_ch: 3
num_feat: 64
input_size: 128
```
