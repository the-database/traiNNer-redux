# Training Guidelines
This page includes a summary of suggested guidelines to follow for training, and best practices based on research and community testing. Note these guidelines may not be optimal for every single architecture, where each may have their own individual quirks.


## Training throughput

On SPAN and SPANF3, `network_g.fused_training: true` composes the three convolution kernels and the skip connection on each forward pass. Only one convolution operates on the image. Autograd still updates the original branch weights, and existing checkpoints load with the same keys. Composition changes floating-point rounding. The option defaults to false.

For repeated paired images, `datasets.train.decoded_cache_mib` retains color-converted images in RAM. Crops, rotations, flips and normalization still run for each sample. The budget applies per worker and per rank. An image larger than the budget follows the regular streaming path. Use persistent workers to retain cached images across epochs.

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

The training log reports completed accumulation windows per second and samples per second. DDP reduces gradients across ranks on the final microbatch of each window. Numbered checkpoints record the consumed sampler position, including at the final update. Older checkpoints resume at the start of their saved epoch. Workers reseed random augmentation when a run resumes.

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
