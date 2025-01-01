# Loss reference
This page lists all available parameters for each loss function in traiNNer-redux. While the default configs use recommended default values and shouldn't need to be modified by most users, advanced users may wish to inspect or modify loss function params to suit their specific use case.
## adistsloss


```yaml
type: adistsloss
window_size: 21
resize_input: false
```
## averageloss


```yaml
type: averageloss
criterion: l1
```
## bicubicloss


```yaml
type: bicubicloss
criterion: l1
```
## charbonnierloss


```yaml
type: charbonnierloss
reduction: mean
eps: 1e-12
```
## colorloss


```yaml
type: colorloss
criterion: l1
```
## contextualloss


```yaml
type: contextualloss
layer_weights: ~
crop_quarter: false
max_1d_size: 100
distance_type: cosine
b: 1.0
band_width: 0.5
use_vgg: true
net: vgg19
calc_type: regular
z_norm: false
```
## distsloss


```yaml
type: distsloss
as_loss: true
load_weights: true
use_input_norm: true
clip_min: 0
```
## ffloss


```yaml
type: ffloss
alpha: 1.0
patch_factor: 1
ave_spectrum: true
log_matrix: false
batch_matrix: false
```
## ganloss


```yaml
type: ganloss
gan_type: vanilla
real_label_val: 1.0
fake_label_val: 0.0
```
## hsluvloss


```yaml
type: hsluvloss
criterion: l1
```
## l1loss


```yaml
type: l1loss
reduction: mean
```
## ldlloss


```yaml
type: ldlloss
criterion: l1
```
## lumaloss


```yaml
type: lumaloss
criterion: l1
```
## mseloss


```yaml
type: mseloss
reduction: mean
```
## mssimloss


```yaml
type: mssimloss
window_size: 11
in_channels: 3
sigma: 1.5
k1: 0.01
k2: 0.03
l: 1
padding: ~
cosim: true
cosim_lambda: 5
```
## msssiml1loss


```yaml
type: msssiml1loss
gaussian_sigmas: ~
data_range: 1.0
k: [0.01, 0.03]
alpha: 0.1
cuda_dev: 0
```
## multiscaleganloss


```yaml
type: multiscaleganloss
real_label_val: 1.0
fake_label_val: 0.0
```
## nccloss


```yaml
type: nccloss
```
## perceptualfp16loss


```yaml
type: perceptualfp16loss
layer_weights: ~
w_lambda: 0.01
alpha: ~
criterion: pd+l1
num_proj_fd: 256
phase_weight_fd: 1.0
stride_fd: 1
```
## perceptualloss


```yaml
type: perceptualloss
layer_weights: ~
w_lambda: 0.01
alpha: ~
criterion: pd+l1
num_proj_fd: 256
phase_weight_fd: 1.0
stride_fd: 1
```
