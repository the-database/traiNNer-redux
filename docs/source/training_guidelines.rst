Training Guidelines
===================

This page includes a summary of suggested guidelines to follow for
training, and best practices based on research and community testing.
Note these guidelines may not be optimal for every single architecture,
where each may have their own individual quirks.

Guidelines
----------

Training Schedule
-----------------

When training from scratch, the general strategy is to train the model
with easy to learn tasks before difficult tasks. So to train a 4x scale
model to handle degradations, a common process is:

1. Train 2x bicubic from scratch:

   -  LR images are simply bicubic downscales of the HR images.
   -  Only MS-SSIM L1 loss is enabled.
   -  The training settings of the official paper models are often a
      safe choice for batch size, crop size, scheduler milestones, and
      total iterations. The training settings are often available on
      their GitHub repo or described in their paper. For example the
      training settings for DAT_2_X2 are
      `here `__:

      -  This model is trained until validation metrics peak, which is
         often at least 500,000 iterations, depending on the
         architecture
      -  With the AdamW optimizer, the learning rate is 2e-4 when
         training from scratch
      -  Larger batch size benefits training from scratch. Gradient
         accumulation can be used to train with a larger effective batch
         size. For example the official DAT 2 training settings use a
         batch size of 8 per GPU on 4 NVIDIA A100 GPUs which is a total
         batch size of 32. A single RTX 4090 does not have enough VRAM
         to train 2x DAT 2 with batch 32, but it can be trained with
         batch 8 and accum_iter 4, for a total effective batch size of
         32.

   -  The ``*_fromscratch.yml`` templates are set up to train with these
      settings
   -  This model is your 2x pretrain

2. Train 4x bicubic:

   -  Use the 2x pretrain from the previous step as a pretrain for this
      4x model, with ``strict_load_g`` set to ``False``
   -  The LR images are bicubic downcsales of the HR images
   -  Only MS-SSIM L1 loss is enabled
   -  The same batch and lq crop size (``lq_size``) from the 2x training
      settings are used for the 4x training settings. The training
      settings of the official paper models are a safe choice for
      scheduler milestones and total iterations.

      -  This model is trained until validation metrics peak, which
         often ranges from 50,000 iterations for lighter architectures
         to 250,000+ iterations for heavier ones
      -  The learning rate is reduced to 1e-4 when finetuning.

   -  This model is your 4x pretrain

3. Train 4x with degadations

   -  Use the 4x pretrain from the previous step as a pretrain for this
      4x model, with ``strict_load_g`` set to ``True``
   -  The LR images are degraded with the types of degradations you want
      the model to handle, such as JPEG or h264
   -  All of the typical losses are enabled for this step: MS-SSIM L1,
      Perceptual, HSLuv, GAN
   -  The learning rate is 1e-4 for the generator and discriminator
      optimizers
   -  This stage of training benefits from larger ``lq_size`` and batch
      size can be reduced to allow larger ``lq_size``
   -  This model is trained until it looks good. You can also look at
      validation metrics, but your eyes should be the final judge
   -  The ``*_finetune.yml`` templates are set up to train with these
      settings
