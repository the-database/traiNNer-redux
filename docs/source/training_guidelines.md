# Training Guidelines
This page includes a summary of suggested guidelines to follow for training, and best practices based on research and community testing. Note these guidelines may not be optimal for every single architecture, where each may have their own individual quirks.


## Training throughput

On SPAN and SPANF3, `network_g.fused_training: true` composes the three convolution kernels and the skip connection on each forward pass. Only one convolution operates on the image. Autograd still updates the original branch weights, and existing checkpoints load with the same keys. Composition changes floating-point rounding. The option defaults to false.

For paired disk images, `datasets.train.decoded_cache_mib` retains color-converted images in RAM. Crops, rotations, flips and normalization still run for each sample. A cache miss decodes the whole cacheable image. The default of zero lets libvips decode what the crop needs. Each worker on each rank has its own cache. With shuffled data, compare each worker's budget to the decoded working set and multiply the budget by the total worker count when estimating RAM. Persistent workers retain their cache across epochs. An individual image larger than the budget follows the streaming path.

The maintainer tested 2,000 image pairs with a decoded footprint of 1,594 MiB, eight workers and batch 16. With caching off, the loader delivered about 52 batches/s. At 64 MiB per worker it fell to 41.8, and at 256 MiB it delivered 38–44. Budgets of 1,024 and 2,048 MiB reached about 72 batches/s. On a larger shuffled corpus, start with caching off and compare on the real data before reserving the RAM. [Maintainer's cache measurements](https://github.com/the-database/traiNNer-redux/pull/339#pullrequestreview-6012332521)

This excerpt shows the settings for the eight-block, 64-feature SPANF3 model in an existing 2x paired-image training config.

```yaml
use_amp: true
amp_bf16: true
use_channels_last: true
use_cuda_graph: true
network_g:
  type: spanf3
  feature_channels: 64
  num_blocks: 8
  unshuffle_mod: true
  fused_training: true
datasets:
  train:
    decoded_cache_mib: 32
```

### RTX 5090 measurements

The following runs used Windows, PyTorch 2.15.0+cu132, BF16, batch 16, 96×96 LR crops, AdamW, Charbonnier loss and CUDA graphs. Each ran through `train.py` for 100 updates, including PNG loading, augmentation, transfers, backward and optimizer updates. Rates cover updates 21–100. All five image pairs fit within 32 MiB per worker. Validation and EMA were disabled.

| SPANF3 training | Decoded cache per worker | First run, LR crops/s | Final build, LR crops/s |
| --- | ---: | ---: | ---: |
| Original branches | Off | 589 | 584 |
| Fused convolution | Off | 593 | 594 |
| Original branches | 32 MiB | 1,309 | 1,336 |
| Fused convolution | 32 MiB | 3,056 | 3,096 |

With data already on the GPU, the same model and crop shape reached 1,356 crops/s with the branches and 3,385 with fusion using L1 loss and CUDA graphs. Shape and execution mode matter. In eager BF16 at batch 8, 64×64 LR crops, five blocks and 48 features, fusion reduced the rate from 514 to 457 crops/s. Cache gains also depend on how often a worker sees the same source image.

As a short quality check, the 100-update checkpoints scored 20.05 dB with the original branches and 20.52 dB with fusion and caching. These are mean RGB PSNR values across three held-out images with bicubic 2x inputs and a two-pixel border crop. Time to a production quality target requires a longer run on the intended corpus.

Run from the repository root in the training environment. The paired benchmark accepts a directory containing matching 2x `hr/` and `lr/` images via `--paired-root`. The GPU-resident benchmark checks that AdamW applied every timed update.

```shell
python -m scripts.benchmarking.benchmark_paired_training
python -m scripts.benchmarking.benchmark_span_training --preset balanced --batch-size 16 --lr-size 96 --cuda-graph
```

The training log reports completed accumulation windows per second and samples per second. Dataloader setup reports microbatches per epoch. Timing includes checkpointing and validation that took place between log entries. ETA estimation starts after the first log window to exclude worker startup, compilation and graph capture during that window. With accumulation enabled, loss and gradient statistics use the sample counts for the specific metric. Existing TensorBoard series for total loss, gradient norm and scaler can change level on resume.

A numbered checkpoint records the last completed window and the per-rank sampler position, including at the final update. On Ctrl+C, a single-GPU trainer finishes its window and saves any progress since the previous checkpoint. DDP ranks agree on stopping at the next log, numbered-save, or final-update boundary. A numbered write failure stops training. If a final `*_latest` file remains locked, the trainer warns and proceeds to final validation using the completed run. A single-process failure at a clean update boundary attempts a recovery save. Failures partway through a window leave the previous numbered checkpoint as the resume point.

Older states resume at the start of their saved epoch. With a fixed manual seed, new sessions restart their augmentation RNG streams. The saved sampler position restores image order, while exact continuation of stochastic crops, flips and degradations would also require their RNG state.

### Training speed across image SR architectures

#### Controlled measurements of the code changes

| Comparison on one RTX 5090 | Reference | Updated | Conditions held fixed |
| --- | ---: | ---: | --- |
| DAT2 window packing, LR crops/s | 31.02 | 34.50 | CUDA graphs, cache, batch 2, LR crop 64 |
| Compact PReLU fusion, LR crops/s | 1,527 | 2,237 | CUDA graphs, cache, batch 2, LR crop 64 |
| Compact PReLU fusion, LR crops/s | 3,663 | 4,786 | CUDA graphs, cache, batch 8, LR crop 64 |
| HAT-L eager, LR crops/s | 7.15 | 7.30 | Graphs off, cache off, batch 2, LR crop 64 |
| HAT-L overlap block, forward and backward ms | 1.84 | 1.48 | Same block, batch 2, LR crop 64 |

#### Combined effect of trainer and settings

These runs used the full registry presets on one RTX 5090, with 2× upscaling, batch 2, 64×64 LR crops, BF16, channels-last tensors, AdamW and Charbonnier loss. Both columns include loading, augmentation, GPU transfer, backward and the optimizer. The reference used commit `285b24f` in eager mode with caching off. The updated configuration enabled CUDA graph replay, already available on master, and a 32 MiB cache in each of four workers. Compact and SuperUltraCompact also enabled `network_g.fused_training`. SPAN-S used its original branches in both columns. The gains for unmodified generators include the effect of enabling those settings.

| Generator | Eager, cache off, LR crops/s | Replay and cache, LR crops/s | Ratio |
| --- | ---: | ---: | ---: |
| HAT-L | 7.15 | 16.47 | 2.30× |
| DAT2 | 9.25 | 34.50 | 3.73× |
| Compact | 339.73 | 2,237.14 | 6.59× |
| SuperUltraCompact | 485.68 | 3,672.69 | 7.56× |
| SwinIR-M | 13.80 | 73.04 | 5.29× |
| ATD | 6.99 | 18.47 | 2.64× |
| DRCT-L | 14.07 | 28.77 | 2.04× |
| ESRGAN | 26.15 | 126.79 | 4.85× |
| RealPLKSR | 44.25 | 143.01 | 3.23× |
| MoSR | 58.98 | 185.10 | 3.14× |
| ArtCNN R8F64 | 284.77 | 1,377.00 | 4.84× |
| SPAN-S | 114.94 | 533.28 | 4.64× |

Each rate is total samples divided by elapsed time across four equal-length windows after warmup. Windows contained 5–30 updates. All five repeatedly sampled PNG pairs fit in each worker's cache. Cache hit rate and batch size matter most for the lightweight models. A fresh compile, validation, checkpoints and EMA are outside the timed windows.

- **HAT.** HAT-L already selects FlexAttention on this GPU. Building the shifted-window mask on the GPU allows the trainer to record the step. The earlier version fell back to eager at capture. Directly gathering overlapping keys and values reduced one HAT-L overlap block's forward plus backward time from 1.84 to 1.48 ms at the same batch and crop. With graph replay off, the updated full model measured 7.30 crops/s.
- **DAT and DAT-iLN.** The spatial branches pack queries, keys and values together. The convolution branches consume channels-last views. With graphs and caching on in both runs, full DAT2 rose from 31.02 to 34.50 crops/s. The iLN variant also passed the reference output and gradient check.
- **Compact, UltraCompact and SuperUltraCompact.** The CUDA training option fuses convolution bias with the per-channel PReLU and its backward reductions. With BF16, the fused path adds bias after the convolution output has been rounded. Early-layer bias and slope gradients can be more sensitive to that rounding. Existing checkpoints retain their convolution and slope parameters. The option defaults to false.
- **SwinIR and ATD.** SwinIR now creates its shifted-window masks on the GPU during capture even when the crop differs from the preset's image size. ATD caches its training mask and restores category-sorted pixels in one batched scatter. Both models passed a replay check with changed input pixels.

Spandrel loaded the saved HAT-L, DAT2 and Compact checkpoints. Mean RGB PSNR on three held-out bicubic image pairs, cropped by two border pixels, was 9.93 → 9.92 dB for HAT-L and 11.32 → 11.22 dB for DAT2 after 25 updates. Compact scored 22.7283 → 22.7288 dB after 150 updates. Measuring time to a target PSNR requires training to that target on the intended dataset.

Run the comparison from the repository root. For a working dataset, pass `--paired-root` with matching `hr/` and `lr/` directories and set `--scale`, `--batch-size` and `--lr-size` to match the training job. The runner checks the training log for successful graph capture and skipped non-finite updates.

```shell
python -m scripts.benchmarking.benchmark_training --arch hat_l dat_2 swinir_m atd --cuda-graph --cache-mib 32
python -m scripts.benchmarking.benchmark_training --arch compact superultracompact --cuda-graph --cache-mib 32 --network-options '{"fused_training":true}' --interval 30
```

## Background
Super resolution methods generally fall into two categories: fidelity oriented and perceptual oriented.

Fidelity oriented super resolution typically uses bicubic degradations for the dataset and only optimizes a pixel loss such as L1. When researchers propose a new super resolution architecture, they use a standard research dataset such as DF2K, train from scratch and evaluate their architecture using the fidelity oriented method, and most offer their pretrained models for download on their original project repository. Most research models are trained on multiple NVIDIA data center GPUs, which allows the use of higher batch such as 32 or 64, and they are trained quickly because multiple GPUs are used in parallel. Since higher batch size results in higher fidelity, it's difficult to match the fidelity of research pretrains on consumer GPUs with limited VRAM on a single GPU.

- Training from scratch on L1 loss is stable as long as the architecture is stable, and the model continues to improve the more it's trained, though there are diminishing returns for training longer.
- Because these models are trained on datasets with bicubic degradations and pixel loss only, they produce blurry upscales which lack convincing textures. But they are faithful to the input image.

Perceptual oriented models tend to use more complex degradations, such as one or more of JPEG compression, blur, or noise, and optimize multiple losses, including a pixel loss, perceptual loss, and adversarial (GAN) loss. When training a perceptual model, a pretrain should always be used for the generator. Fidelity oriented models, like the official research models, are well suited for this purpose.

- Training with multiple losses including GAN loss is much less stable, and training can collapse when trained for too long.
- Perceptual oriented models are able to restore fine details on degraded images, but they may trade some faithfulness to the input image.

Most users find perceptual oriented models more practical and only ever train perceptual oriented models.

## Training a Perceptual Model with an Existing Pretrain

When training a perceptual model, a pretrain model should always be used for the generator. While any model can be used for your pretrain, official research models are often a good choice. When you use a pretrain model, you risk introducing any flaws of the pretrain model into your own model. Official research models are likely to have the least amount of flaws, and they have high fidelity to give your model a good starting point.

1. Train a perceptual model with degradations:
   - Download an official pretrain for the architecture you want to train, and move it to `experiments/pretrained_models`
   - Open the `*_gan.yml` template and set the `pretrain_network_g` path to the pretrain you downloaded

## Training a Perceptual Model from Scratch

While using an existing pretrain is recommended when training a perceptual model, you might prefer to train a model completely from scratch, including training your own pretrain with your own dataset. When training from scratch, the general strategy is to train the model with easy to learn tasks before difficult tasks. So to train a 4x scale model to handle degradations, a common process is:

1. Train 2x bicubic from scratch:
   - LR images are simply bicubic downscales of the HR images.
   - Only Charbonnier loss is enabled.
   - The training settings of the official paper models are often a safe choice for batch size, crop size, scheduler milestones, and total iterations. The training settings are often available on their GitHub repo or described in their paper. For example the training settings for DAT_2_X2 are [here](https://github.com/zhengchen1999/DAT/blob/main/options/Train/train_DAT_2_x2.yml):
      - This model is trained until distortion validation metrics (PSNR and SSIM) peak, which is often at least 500,000 iterations, depending on the architecture
      - With the AdamW optimizer, the learning rate is 2e-4 when training from scratch
      - Larger batch size benefits training from scratch. Gradient accumulation can be used to train with a larger effective batch size. For example the official DAT 2 training settings use a batch size of 8 per GPU on 4 NVIDIA A100 GPUs which is a total batch size of 32. A single RTX 4090 does not have enough VRAM to train 2x DAT 2 with batch 32, but it can be trained with batch 8 and accum_iter 4, for a total effective batch size of 32.
   - The `*_fidelity.yml` templates are set up to train with these settings
   - This model is your 2x fidelity pretrain
2. Train 4x bicubic:
   - Use the 2x pretrain from the previous step as a pretrain for this 4x model, with `strict_load_g` set to `False`
   - The LR images are bicubic downcsales of the HR images
   - Only Charbonnier loss is enabled
   - The same batch and lq crop size (`lq_size`) from the 2x training settings are used for the 4x training settings. The training settings of the official paper models are a safe choice for scheduler milestones and total iterations. The milestones and total iterations are usually cut in half from the previous step.
       - This model is trained until distortion validation metrics (PSNR and SSIM) peak, which is often at least 250,000 iterations, depending on the architecture.
       - The learning rate typically is reduced to 1e-4 when finetuning.
   - This model is your 4x fidelity pretrain

3. Train 4x with degadations
   - Use the 4x pretrain from the previous step as a pretrain for this 4x model, with `strict_load_g` set to `True`
   - The LR images are degraded with the types of degradations you want the model to handle, such as JPEG or h264
   - All of the typical losses are enabled for this step: MSSIM, Perceptual, HSLuv, GAN
   - The learning rate is 1e-4 for the generator and discriminator optimizers
   - This stage of training benefits from larger `lq_size` and batch size can be reduced to allow larger `lq_size`
   - This model is trained until it looks good. Distortion validation metrics (PSNR and SSIM) are not a good indicator of good perceptual quality. Perceptual validation metrics (TOPIQ, LPIPS, DISTS) are correlated with human perception of visual quality to some extent, but your eyes should be the final judge
   - The `*_gan.yml` templates are set up to train with these settings
