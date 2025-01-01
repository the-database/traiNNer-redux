# Config file reference
## Top level options

### name

  Name of the experiment. It should be a unique name if you want to run a new experiment. If you enable auto resume, the experiment with this name will be resumed instead of starting a new training run.

  Type: str
### scale

  Scale of the model. Most architectures support a scale of 1, 2, 3, 4, or 8. A scale of 1 can be used for restoration models that don't change the resolution of the input image. A scale of 2 means the width and height of the input image are doubled, so a 640x480 input will be upscaled to 1280x960.

  Type: int
### num_gpu

  The number of GPUs to use for training, if using multiple GPUs.

  Type: Literal, int

### network_g

  The options for the generator model.

  Type: dict
### network_d

  The options for the discriminator model.

  Type: dict
### manual_seed

  Deterministic mode, slows down training. Only use for reproducible experiments.

  Type: int









### use_amp

  Speed up training and reduce VRAM usage. NVIDIA only.

  Type: bool
### amp_bf16

  Use bf16 instead of fp16 for AMP, RTX 3000 series or newer only. Only recommended if fp16 doesn't work.

  Type: bool
### use_channels_last

  Enable channels last memory format while using AMP. Reduces VRAM and speeds up training for most architectures, but some architectures are slower with channels last.

  Type: bool
### fast_matmul

  Trade precision for performance.

  Type: bool
### use_compile

  Enable torch.compile for the generator model, which takes time on startup to compile the model, but can speed up training after the model is compiled. However, compilation must be redone when starting training each time, as the compiled model is not saved, so for models that take too long to compile it may not worth it.

  Type: bool
### detect_anomaly

  Whether or not to enable anomaly detection, which can be useful for debugging NaNs that occur during training. Has a significant performance hit and should be disabled when not debugging.

  Type: bool
### high_order_degradation

  Whether or not to enable OTF (on the fly) degradations, which generates LRs on the fly.

  Type: bool
### high_order_degradations_debug

  Whether or not to enable debugging for OTF, which saves the OTF generated LR images so they can be inspected to view the effect of different OTF settings.

  Type: bool
### high_order_degradations_debug_limit

  The maximum number of OTF images to save when debugging is enabled.

  Type: int
### dataroot_lq_prob

  Probability of using paired LR data instead of OTF LR data.

  Type: float
### lq_usm

  Whether to enable unsharp mask on the LQ image.

  Type: bool
### lq_usm_radius_range

  For the unsharp mask of the LQ image, use a radius randomly selected from this range.

  Type: tuple
### blur_prob

  Probability of applying the first blur to the LQ, between 0 and 1.

  Type: float
### resize_prob

  List of 3 probabilities for the first resize which should add up to 1: the probability of upscaling, the probability of downscaling, and the probability of no resize.

  Type: list
### resize_mode_list

  List of possible resize modes to use for the first resize.

  Type: list
### resize_mode_prob

  List of probabilities for the first resize of selecting the corresponding resize mode in `resize_mode_list`.

  Type: list
### resize_range

  The resize range for the first resize, in the format `[min_resize, max_resize]`.

  Type: tuple
### gaussian_noise_prob

  The probability of applying the first gaussian noise to the LQ, between 0 and 1.

  Type: float



### jpeg_prob

  The probability of applying the first JPEG degradation to the LQ, between 0 and 1.

  Type: float
### jpeg_range

  The range of JPEG quality to apply for the first JPEG degradation, in the format `[min_quality, max_quality]`.

  Type: tuple





















## Dataset options (`datasets.train` and `datasets.val`)

### name

  Name of the dataset. It should be unique compared to other datasets in this config, but the exact name isn't very important.

  Type: str


### num_worker_per_gpu

  Number of subprocesses to use for data loading with PyTorch dataloader.

  Type: int
### batch_size_per_gpu

  Increasing stabilizes training but going too high can cause issues. Use multiple of 8 for best performance with AMP. A higher batch size, like 32 or 64 is more important when training from scratch, while smaller batches like 8 can be used when training with a quality pretrain model.

  Type: int
### accum_iter

  Using values larger than 1 simulates higher batch size by trading performance for reduced VRAM usage. If accum_iter = 4 and batch_size_per_gpu = 6 then effective batch size = 4 * 6 = 24 but performance may be as much as 4 times as slow.

  Type: int
### use_hflip

  Randomly flip the images horizontally.

  Type: bool
### use_rot

  Randomly rotate the images.

  Type: bool



### lq_size

  During training, a square of this size is cropped from LR images. Larger is usually better but uses more VRAM. Previously gt_size, use lq_size = gt_size / scale to convert. Use multiple of 8 for best performance with AMP.

  Type: int



### dataset_enlarge_ratio

  Increase if the dataset is less than 1000 images to avoid slowdowns. Auto will automatically enlarge small datasets only.

  Type: Literal, int





### clip_size

  Number of frames per clip in `PairedVideoDataset`. Must match the `clip_size` option for video generator networks such as `tscunet`.

  Type: int
### dataroot_gt

  Path to the HR (high res) images in your training dataset. Specify one or multiple folders, separated by commas.

  Type: str, list
### dataroot_lq

  Path to the LR (low res) images in your training dataset. Specify one or multiple folders, separated by commas.

  Type: str, list

### filename_tmpl

  Filename template to use for LR images. Commonly used values might be `{}x2` or `{}x4`, which should be used if the LR dataset filename is in the format filename.png while the LR dataset filename is in the format `filename_x2.png` or `filename_x4.png`. This is common on some research datasets such as DIV2K or DF2K.

  Type: str


















## Path options (`path`)











### strict_load_g

  Whether to load the pretrain model for the generator in strict mode. It should be enabled in most cases, unless you want to partially load a pretrain of a different scale or with slightly different hyperparameters.

  Type: bool




### strict_load_d

  Whether to load the pretrain model for the discriminator in strict mode. It should be enabled in most cases.

  Type: bool

## Train options (`train`)

### total_iter

  The total number of iterations to train.

  Type: int
### optim_g

  The optimizer to use for the generator model.

  Type: dict
### ema_decay

  The decay factor to use for EMA (exponential moving average). Set to 0 to disable EMA.

  Type: float
### grad_clip

  Whether or not to enable gradient clipping, which can improve stability when using higher learning rates, but can also cause issues in some situations.

  Type: bool
### warmup_iter

  Gradually ramp up learning rates until this iteration, to stabilize early training. Use -1 to disable.

  Type: int
### scheduler

  Options for the optimizer scheduler. If there are multiple optimizers, both will use the same scheduler options.

  Type: SchedulerOptions
### optim_d

  The optimizer to use for the discriminator model.

  Type: dict
### losses

  The list of loss functions to optimize.

  Type: list


















### use_moa

  Whether to enable mixture of augmentations, which augments the dataset on the fly to create more variety and help the model generalize.

  Type: bool
### moa_augs

  The list of augmentations to choose from, only one is selected per iteration.

  Type: list
### moa_probs

  The probability each augmentation in moa_augs will be applied. Total should add up to 1.

  Type: list
### moa_debug

  Save images before and after augment to debug/moa folder inside of the root training directory.

  Type: bool
### moa_debug_limit

  The max number of iterations to save augmentation images for.

  Type: int
## Scheduler options (`train.scheduler`)

### type

  Name of the optimizer scheduler to use for all optimizers. For a list of scheduler names, see the [PyTorch documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).

  Type: str
### milestones

  List of milestones, iterations where the learning rate is reduced.

  Type: list
### gamma

  At each milestone, the learning rate is multiplied by this number, so a gamma of 0.5 cuts the learning rate in half at each milestone.

  Type: float
## Validation options (`val`)

### val_enabled

  Whether to enable validations. If disabled, all validation settings below are ignored.

  Type: bool
### save_img

  Whether to save the validation images during validation, in the experiments/<name>/visualization folder.

  Type: bool
### val_freq

  How often to run validations, in iterations.

  Type: int
### suffix

  Optional suffix to append to saved filenames.

  Type: str
### metrics_enabled

  Whether to run metrics calculations during validation.

  Type: bool


## Logging options (`logger`)

### print_freq

  How often to print logs to the console, in iterations.

  Type: int
### save_checkpoint_freq

  How often to save model checkpoints and training states, in iterations.

  Type: int
### use_tb_logger

  Whether or not to enable TensorBoard logging.

  Type: bool
### save_checkpoint_format

  Format to save model checkpoints.

  Options: safetensors, pth

