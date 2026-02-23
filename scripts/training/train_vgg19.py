#!/usr/bin/env python3
"""
Modern VGG19 training script for perceptual loss backbone.

Based on torchvision's classification training reference, modernized and adapted
for training VGG19 (standard or antialiased) from scratch on ImageNet.

Usage:
    # Standard VGG19
    python train_vgg19.py --data-path /path/to/imagenet --model vgg19

    # Antialiased VGG19
    python train_vgg19.py --data-path /path/to/imagenet --model vgg19 --antialiased

    # Resume training
    python train_vgg19.py --data-path /path/to/imagenet --model vgg19 --resume checkpoint.pth

    # Distributed training
    torchrun --nproc_per_node=8 train_vgg19.py --data-path /path/to/imagenet --model vgg19
"""

import argparse
import datetime
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.utils.data
import torchvision
import torchvision.transforms.v2 as transforms
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from torchvision.transforms import InterpolationMode

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    data_path: str = ""
    train_crop_size: int = 224
    val_resize_size: int = 256
    val_crop_size: int = 224
    interpolation: str = "bilinear"
    workers: int = 16

    # Model
    model: str = "vgg19"
    antialiased: bool = False
    filter_size: int = 4
    replicate_padding: bool = False
    use_normalization: bool = True

    # Training
    batch_size: int = 32
    epochs: int = 90
    start_epoch: int = 0

    # Optimizer
    opt: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    norm_weight_decay: float | None = None

    # LR scheduler
    lr_scheduler: str = "steplr"
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    lr_min: float = 0.0
    lr_warmup_epochs: int = 0
    lr_warmup_method: str = "linear"
    lr_warmup_decay: float = 0.01

    # Regularization
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    random_erase: float = 0.0
    auto_augment: str | None = None

    # AMP / precision
    amp: bool = False
    amp_dtype: str = "float16"  # float16 or bfloat16
    clip_grad_norm: float | None = None
    tf32: bool = True
    matmul_precision: str = "high"  # highest, high, medium

    # torch.compile
    compile: bool = False
    compile_mode: str = "reduce-overhead"  # default, reduce-overhead, max-autotune

    # EMA
    model_ema: bool = False
    model_ema_steps: int = 32
    model_ema_decay: float = 0.99998

    # Misc
    device: str = "cuda"
    output_dir: str = "."
    resume: str = ""
    test_only: bool = False
    skip_val: bool = False
    print_freq: int = 10
    save_freq: int = 1
    deterministic: bool = False
    sync_bn: bool = False

    # Distributed (set at runtime)
    distributed: bool = field(default=False, init=False)
    rank: int = field(default=0, init=False)
    world_size: int = field(default=1, init=False)
    local_rank: int = field(default=0, init=False)
    gpu: int = field(default=0, init=False)


# =============================================================================
# Antialiased VGG19
# =============================================================================


class BlurPool2d(nn.Module):
    """Blur pooling layer for antialiased downsampling.

    From "Making Convolutional Networks Shift-Invariant Again" (Zhang 2019).
    Applies a low-pass filter before subsampling to reduce aliasing.
    """

    def __init__(
        self,
        channels: int,
        filter_size: int = 4,
        stride: int = 2,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.filter_size = filter_size
        self.stride = stride

        if padding is None:
            padding = (filter_size - 1) // 2

        self.padding = padding

        # Create blur kernel
        kernel = self._get_blur_kernel(filter_size)
        kernel = kernel[None, None, :, :].repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    @staticmethod
    def _get_blur_kernel(size: int) -> Tensor:
        """Get Pascal's triangle-based blur kernel."""
        if size == 1:
            return torch.tensor([1.0])
        if size == 2:
            kernel_1d = torch.tensor([1.0, 1.0])
        elif size == 3:
            kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        elif size == 4:
            kernel_1d = torch.tensor([1.0, 3.0, 3.0, 1.0])
        elif size == 5:
            kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        elif size == 6:
            kernel_1d = torch.tensor([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif size == 7:
            kernel_1d = torch.tensor([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        else:
            raise ValueError(f"Unsupported filter size: {size}")

        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.conv2d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=self.channels,
        )


def make_vgg_layers(
    cfg: list[int | str],
    batch_norm: bool = False,
    antialiased: bool = False,
    filter_size: int = 4,
    replicate_padding: bool = False,
) -> nn.Sequential:
    """Build VGG feature layers.

    Args:
        cfg: Layer configuration (channel counts and 'M' for maxpool).
        batch_norm: Whether to use batch normalization.
        antialiased: Whether to use antialiased pooling.
        filter_size: Filter size for antialiased pooling.
        replicate_padding: Use replicate padding on first conv (matches perceptual loss).

    Returns:
        Sequential module with VGG layers.
    """
    layers: list[nn.Module] = []
    in_channels = 3
    is_first_conv = True

    for v in cfg:
        if v == "M":
            if antialiased:
                # MaxPool with stride=1 (no downsampling) + BlurPool for downsampling
                layers.append(nn.MaxPool2d(kernel_size=2, stride=1, padding=1))
                layers.append(
                    BlurPool2d(in_channels, filter_size=filter_size, stride=2)
                )
            else:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            assert isinstance(v, int)
            padding_mode = (
                "replicate" if (is_first_conv and replicate_padding) else "zeros"
            )
            conv = nn.Conv2d(
                in_channels, v, kernel_size=3, padding=1, padding_mode=padding_mode
            )
            if batch_norm:
                layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv, nn.ReLU(inplace=True)])
            in_channels = v
            is_first_conv = False

    return nn.Sequential(*layers)


# VGG19 configuration
VGG19_CFG: list[int | str] = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    512,
    "M",
]


class VGG19(nn.Module):
    """VGG19 model with optional antialiased pooling."""

    def __init__(
        self,
        num_classes: int = 1000,
        batch_norm: bool = False,
        antialiased: bool = False,
        filter_size: int = 4,
        dropout: float = 0.5,
        replicate_padding: bool = False,
    ) -> None:
        super().__init__()

        self.features = make_vgg_layers(
            VGG19_CFG,
            batch_norm=batch_norm,
            antialiased=antialiased,
            filter_size=filter_size,
            replicate_padding=replicate_padding,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(config: TrainConfig, num_classes: int) -> nn.Module:
    """Create model based on config."""
    if config.model == "vgg19":
        return VGG19(
            num_classes=num_classes,
            batch_norm=False,
            antialiased=config.antialiased,
            filter_size=config.filter_size,
            replicate_padding=config.replicate_padding,
        )
    elif config.model == "vgg19_bn":
        return VGG19(
            num_classes=num_classes,
            batch_norm=True,
            antialiased=config.antialiased,
            filter_size=config.filter_size,
            replicate_padding=config.replicate_padding,
        )
    else:
        raise ValueError(f"Unknown model: {config.model}. Use 'vgg19' or 'vgg19_bn'.")


# =============================================================================
# Data Loading
# =============================================================================


def get_train_transforms(config: TrainConfig) -> transforms.Compose:
    """Get training transforms."""
    interpolation = InterpolationMode(config.interpolation)

    trans: list[Any] = [
        transforms.RandomResizedCrop(
            config.train_crop_size,
            interpolation=interpolation,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(),
    ]

    # Auto augment
    if config.auto_augment is not None:
        if config.auto_augment == "ra":
            trans.append(transforms.RandAugment(interpolation=interpolation))
        elif config.auto_augment == "ta_wide":
            trans.append(transforms.TrivialAugmentWide(interpolation=interpolation))
        elif config.auto_augment == "augmix":
            trans.append(transforms.AugMix(interpolation=interpolation))
        else:
            trans.append(
                transforms.AutoAugment(
                    policy=transforms.AutoAugmentPolicy(config.auto_augment)
                )
            )

    trans.extend(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    # ImageNet normalization (optional - disable for perceptual loss compatibility)
    if config.use_normalization:
        trans.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    # Random erasing
    if config.random_erase > 0:
        trans.append(transforms.RandomErasing(p=config.random_erase))

    return transforms.Compose(trans)


def get_val_transforms(config: TrainConfig) -> transforms.Compose:
    """Get validation transforms."""
    interpolation = InterpolationMode(config.interpolation)

    trans: list[Any] = [
        transforms.Resize(
            config.val_resize_size, interpolation=interpolation, antialias=True
        ),
        transforms.CenterCrop(config.val_crop_size),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]

    # ImageNet normalization (optional - disable for perceptual loss compatibility)
    if config.use_normalization:
        trans.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    return transforms.Compose(trans)


def get_mixup_cutmix(
    config: TrainConfig, num_classes: int
) -> transforms.MixUp | transforms.CutMix | None:
    """Get mixup/cutmix transform if configured."""
    mixup_transforms: list[Any] = []

    if config.mixup_alpha > 0.0:
        mixup_transforms.append(
            transforms.MixUp(alpha=config.mixup_alpha, num_classes=num_classes)
        )
    if config.cutmix_alpha > 0.0:
        mixup_transforms.append(
            transforms.CutMix(alpha=config.cutmix_alpha, num_classes=num_classes)
        )

    if not mixup_transforms:
        return None
    if len(mixup_transforms) == 1:
        return mixup_transforms[0]
    return transforms.RandomChoice(mixup_transforms)


def load_data(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, int]:
    """Load datasets and create data loaders."""
    # train_dir = Path(config.data_path)  # / "train"
    val_dir = Path(config.data_path)  # / "val"

    # assert train_dir.exists(), f"Training directory not found: {train_dir}"
    assert val_dir.exists(), f"Validation directory not found: {val_dir}"

    # print("Loading training data...")
    # train_dataset = torchvision.datasets.ImageFolder(
    #     str(train_dir),
    #     transform=get_train_transforms(config),
    # )

    print("Loading validation data...")
    val_dataset = torchvision.datasets.ImageFolder(
        str(val_dir),
        transform=get_val_transforms(config),
    )

    # num_classes = len(train_dataset.classes)
    # print(f"Found {num_classes} classes")

    # Create samplers
    if config.distributed:
        # train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        # train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    # Mixup/cutmix collate function
    # mixup_cutmix = get_mixup_cutmix(config, num_classes)
    # if mixup_cutmix is not None:

    #     def collate_fn(batch: list) -> tuple[Tensor, Tensor]:
    #         return mixup_cutmix(*torch.utils.data.default_collate(batch))
    # else:
    #     collate_fn = None

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config.batch_size,
    #     sampler=train_sampler,
    #     num_workers=config.workers,
    #     pin_memory=True,
    #     collate_fn=collate_fn,
    #     persistent_workers=config.workers > 0,
    #     prefetch_factor=2 if config.workers > 0 else None,
    # )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
        prefetch_factor=2 if config.workers > 0 else None,
    )

    return None, val_loader, 1000


# =============================================================================
# Training Utilities
# =============================================================================


class ExponentialMovingAverage:
    """Maintains exponential moving average of model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        device: torch.device,
    ) -> None:
        self.decay = decay
        self.num_updates = 0
        self.shadow_params = {
            name: param.clone().detach().to(device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name].sub_(
                        (1.0 - decay) * (self.shadow_params[name] - param)
                    )

    def apply(self, model: nn.Module) -> dict[str, Tensor]:
        """Apply EMA weights to model, return original weights for restoration."""
        original = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    original[name] = param.clone()
                    param.copy_(self.shadow_params[name])
        return original

    def restore(self, model: nn.Module, original: dict[str, Tensor]) -> None:
        """Restore original weights after EMA evaluation."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original:
                    param.copy_(original[name])

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = state_dict["shadow_params"]


class MetricTracker:
    """Tracks and averages metrics during training."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.sum:
            self.sum[name] = 0.0
            self.count[name] = 0
        self.sum[name] += value * n
        self.count[name] += n

    def avg(self, name: str) -> float:
        assert name in self.sum and self.count[name] > 0, (
            f"No values recorded for {name}"
        )
        return self.sum[name] / self.count[name]

    def sync_distributed(self) -> None:
        """Synchronize metrics across distributed processes."""
        if not dist.is_initialized():
            return

        for name in self.sum:
            sum_tensor = torch.tensor(self.sum[name], device="cuda")
            count_tensor = torch.tensor(self.count[name], device="cuda")

            dist.all_reduce(sum_tensor)
            dist.all_reduce(count_tensor)

            self.sum[name] = sum_tensor.item()
            self.count[name] = int(count_tensor.item())


def accuracy(
    output: Tensor, target: Tensor, topk: tuple[int, ...] = (1,)
) -> list[Tensor]:
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # Handle soft targets (mixup/cutmix)
        if target.dim() > 1:
            target = target.argmax(dim=1)

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def set_weight_decay(
    model: nn.Module,
    weight_decay: float,
    norm_weight_decay: float | None = None,
) -> list[dict[str, Any]]:
    """Set weight decay, optionally different for norm layers."""
    norm_classes = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )

    params: dict[str, list[nn.Parameter]] = {"decay": [], "no_decay": []}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter belongs to a norm layer
        is_norm = False
        for module_name, module in model.named_modules():
            if isinstance(module, norm_classes):
                for param_name, _ in module.named_parameters(recurse=False):
                    if name == f"{module_name}.{param_name}":
                        is_norm = True
                        break
            if is_norm:
                break

        if is_norm and norm_weight_decay is not None:
            params["no_decay"].append(param)
        else:
            params["decay"].append(param)

    param_groups = [{"params": params["decay"], "weight_decay": weight_decay}]
    if params["no_decay"]:
        wd = 0.0 if norm_weight_decay is None else norm_weight_decay
        param_groups.append({"params": params["no_decay"], "weight_decay": wd})

    return param_groups


# =============================================================================
# Training Loop
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
    model_ema: ExponentialMovingAverage | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> MetricTracker:
    """Train for one epoch."""
    model.train()
    metrics = MetricTracker()

    num_batches = len(data_loader)
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        batch_start = time.time()

        images = images.to(device, non_blocking=True).to(
            memory_format=torch.channels_last
        )
        targets = targets.to(device, non_blocking=True)

        # Forward pass with optional AMP
        amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
        with torch.amp.autocast("cuda", enabled=scaler is not None, dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if config.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()

        # Update EMA
        if model_ema is not None and batch_idx % config.model_ema_steps == 0:
            model_ema.update(model)

        # Compute accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)

        # Update metrics
        metrics.update("loss", loss.item(), batch_size)
        metrics.update("acc1", acc1.item(), batch_size)
        metrics.update("acc5", acc5.item(), batch_size)
        metrics.update("lr", optimizer.param_groups[0]["lr"], 1)
        metrics.update("img/s", batch_size / (time.time() - batch_start), 1)

        # Log progress
        if batch_idx % config.print_freq == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)

            print(
                f"Epoch [{epoch}][{batch_idx}/{num_batches}]  "
                f"Loss: {metrics.avg('loss'):.4f}  "
                f"Acc@1: {metrics.avg('acc1'):.2f}  "
                f"Acc@5: {metrics.avg('acc5'):.2f}  "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}  "
                f"Img/s: {metrics.avg('img/s'):.1f}  "
                f"ETA: {datetime.timedelta(seconds=int(eta))}"
            )

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    log_suffix: str = "",
) -> MetricTracker:
    """Evaluate model on validation set."""
    model.eval()
    metrics = MetricTracker()

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True).to(
            memory_format=torch.channels_last
        )
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)

        metrics.update("loss", loss.item(), batch_size)
        metrics.update("acc1", acc1.item(), batch_size)
        metrics.update("acc5", acc5.item(), batch_size)

    # Sync across distributed processes
    metrics.sync_distributed()

    suffix = f" {log_suffix}" if log_suffix else ""
    print(
        f"Test{suffix}:  Acc@1: {metrics.avg('acc1'):.2f}  Acc@5: {metrics.avg('acc5'):.2f}"
    )

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: TrainConfig,
    model_ema: ExponentialMovingAverage | None = None,
    scaler: torch.amp.GradScaler | None = None,
    is_best: bool = False,
) -> None:
    """Save training checkpoint."""
    if config.distributed and config.rank != 0:
        return

    # Get model without DDP wrapper
    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "config": config,
    }

    if model_ema is not None:
        checkpoint["model_ema"] = model_ema.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save epoch checkpoint
    torch.save(checkpoint, output_dir / f"model_{epoch}.pth")

    # Save latest checkpoint (overwrite)
    torch.save(checkpoint, output_dir / "checkpoint.pth")

    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, output_dir / "best.pth")


def extract_features_weights(checkpoint_path: str, output_path: str) -> None:
    """Extract just the features (backbone) weights for use in perceptual loss.

    This saves only the convolutional backbone, excluding the classifier.
    The output can be loaded directly into your perceptual loss VGG backbone.

    Args:
        checkpoint_path: Path to training checkpoint (.pth file).
        output_path: Path to save extracted features weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint["model"]

    # Extract only features.* keys and strip the prefix
    features_state = {
        k.replace("features.", ""): v
        for k, v in model_state.items()
        if k.startswith("features.")
    }

    torch.save(features_state, output_path)
    print(f"Extracted {len(features_state)} feature layers to {output_path}")


# =============================================================================
# Main
# =============================================================================


def init_distributed(config: TrainConfig) -> None:
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        config.rank = int(os.environ["RANK"])
        config.world_size = int(os.environ["WORLD_SIZE"])
        config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        config.distributed = True
    elif "SLURM_PROCID" in os.environ:
        config.rank = int(os.environ["SLURM_PROCID"])
        config.local_rank = config.rank % torch.cuda.device_count()
        config.world_size = int(os.environ["SLURM_NTASKS"])
        config.distributed = True
    else:
        config.distributed = False
        return

    torch.cuda.set_device(config.local_rank)
    config.gpu = config.local_rank

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=config.world_size,
        rank=config.rank,
    )
    dist.barrier()


def main(config: TrainConfig) -> None:
    """Main training function."""
    init_distributed(config)

    if config.distributed:
        print(f"Distributed training: rank {config.rank}/{config.world_size}")

    device = torch.device(config.device)

    # Determinism
    if config.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # TF32 and matmul precision
    torch.backends.cuda.matmul.allow_tf32 = config.tf32
    torch.backends.cudnn.allow_tf32 = config.tf32
    torch.set_float32_matmul_precision(config.matmul_precision)

    # Load data
    train_loader, val_loader, num_classes = load_data(config)

    # Create model
    print(
        f"Creating model: {config.model} "
        f"(antialiased={config.antialiased}, "
        f"replicate_padding={config.replicate_padding}, "
        f"normalization={config.use_normalization})"
    )
    model = create_model(config, num_classes)
    model.to(device)
    model.to(memory_format=torch.channels_last)

    if config.distributed and config.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # Optimizer
    parameters = set_weight_decay(model, config.weight_decay, config.norm_weight_decay)

    if config.opt.lower() == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.opt.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.opt}")

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if config.amp else None

    # LR scheduler
    if config.lr_scheduler.lower() == "steplr":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma,
        )
    elif config.lr_scheduler.lower() == "cosineannealinglr":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs - config.lr_warmup_epochs,
            eta_min=config.lr_min,
        )
    else:
        raise ValueError(f"Unknown LR scheduler: {config.lr_scheduler}")

    # Warmup scheduler
    if config.lr_warmup_epochs > 0:
        if config.lr_warmup_method == "linear":
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.lr_warmup_decay,
                total_iters=config.lr_warmup_epochs,
            )
        elif config.lr_warmup_method == "constant":
            warmup_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=config.lr_warmup_decay,
                total_iters=config.lr_warmup_epochs,
            )
        else:
            raise ValueError(f"Unknown warmup method: {config.lr_warmup_method}")

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.lr_warmup_epochs],
        )
    else:
        lr_scheduler = main_scheduler

    # Distributed model
    model_without_ddp = model
    if config.distributed:
        model = DistributedDataParallel(model, device_ids=[config.gpu])
        model_without_ddp = model.module

    # torch.compile
    if config.compile:
        print(f"Compiling model with mode={config.compile_mode}")
        model = torch.compile(model, mode=config.compile_mode)

    # EMA
    model_ema: ExponentialMovingAverage | None = None
    if config.model_ema:
        adjust = (
            config.world_size
            * config.batch_size
            * config.model_ema_steps
            / config.epochs
        )
        alpha = 1.0 - config.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(
            model_without_ddp, decay=1.0 - alpha, device=device
        )

    # Resume from checkpoint
    if config.resume:
        print(f"Resuming from checkpoint: {config.resume}")
        checkpoint = torch.load(config.resume, map_location="cpu", weights_only=False)

        model_without_ddp.load_state_dict(checkpoint["model"])

        if not config.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        config.start_epoch = checkpoint["epoch"] + 1

        if model_ema is not None and "model_ema" in checkpoint:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    # Test only mode
    if config.test_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if model_ema is not None:
            original_weights = model_ema.apply(model_without_ddp)
            evaluate(model, criterion, val_loader, device, config, log_suffix="EMA")
            model_ema.restore(model_without_ddp, original_weights)
        else:
            evaluate(model, criterion, val_loader, device, config)
        return

    # Training loop
    print("Starting training...")
    start_time = time.time()
    best_acc1 = 0.0

    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            device,
            epoch,
            config,
            model_ema,
            scaler,
        )
        lr_scheduler.step()

        # Evaluate (unless skipped)
        if not config.skip_val:
            val_metrics = evaluate(model, criterion, val_loader, device, config)

            if model_ema is not None:
                original_weights = model_ema.apply(model_without_ddp)
                ema_metrics = evaluate(
                    model, criterion, val_loader, device, config, log_suffix="EMA"
                )
                model_ema.restore(model_without_ddp, original_weights)
                current_acc1 = ema_metrics.avg("acc1")
            else:
                current_acc1 = val_metrics.avg("acc1")

            # Save checkpoint
            is_best = current_acc1 > best_acc1
            best_acc1 = max(best_acc1, current_acc1)
        else:
            is_best = False

        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0 or epoch == config.epochs - 1:
            save_checkpoint(
                model,
                optimizer,
                lr_scheduler,
                epoch,
                config,
                model_ema,
                scaler,
                is_best=is_best,
            )

    total_time = time.time() - start_time
    print(f"Training completed in {datetime.timedelta(seconds=int(total_time))}")
    print(f"Best Acc@1: {best_acc1:.2f}")


def parse_args() -> TrainConfig:
    """Parse command line arguments into TrainConfig."""
    parser = argparse.ArgumentParser(
        description="Train VGG19 for perceptual loss backbone"
    )

    # Data
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to ImageNet dataset"
    )
    parser.add_argument("--train-crop-size", type=int, default=224)
    parser.add_argument("--val-resize-size", type=int, default=256)
    parser.add_argument("--val-crop-size", type=int, default=224)
    parser.add_argument("--interpolation", type=str, default="bilinear")
    parser.add_argument("-j", "--workers", type=int, default=16)

    # Model
    parser.add_argument(
        "--model", type=str, default="vgg19", choices=["vgg19", "vgg19_bn"]
    )
    parser.add_argument(
        "--antialiased", action="store_true", help="Use antialiased pooling"
    )
    parser.add_argument(
        "--filter-size", type=int, default=4, help="BlurPool filter size"
    )
    parser.add_argument(
        "--replicate-padding",
        action="store_true",
        help="Use replicate padding on first conv",
    )
    parser.add_argument(
        "--no-normalization", action="store_true", help="Disable ImageNet normalization"
    )

    # Training
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--start-epoch", type=int, default=0)

    # Optimizer
    parser.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--norm-weight-decay", type=float, default=None)

    # LR scheduler
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="steplr",
        choices=["steplr", "cosineannealinglr"],
    )
    parser.add_argument("--lr-step-size", type=int, default=30)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--lr-min", type=float, default=0.0)
    parser.add_argument("--lr-warmup-epochs", type=int, default=0)
    parser.add_argument(
        "--lr-warmup-method", type=str, default="linear", choices=["linear", "constant"]
    )
    parser.add_argument("--lr-warmup-decay", type=float, default=0.01)

    # Regularization
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--random-erase", type=float, default=0.0)
    parser.add_argument("--auto-augment", type=str, default=None)

    # AMP / precision
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"]
    )
    parser.add_argument("--clip-grad-norm", type=float, default=None)
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32")
    parser.add_argument(
        "--matmul-precision",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
    )

    # torch.compile
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )

    # EMA
    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--model-ema-steps", type=int, default=32)
    parser.add_argument("--model-ema-decay", type=float, default=0.99998)

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument(
        "--skip-val", action="store_true", help="Skip validation during training"
    )
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument(
        "--save-freq", type=int, default=1, help="Save checkpoint every N epochs"
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--sync-bn", action="store_true")

    args = parser.parse_args()

    # Build config
    config = TrainConfig(
        data_path=args.data_path,
        train_crop_size=args.train_crop_size,
        val_resize_size=args.val_resize_size,
        val_crop_size=args.val_crop_size,
        interpolation=args.interpolation,
        workers=args.workers,
        model=args.model,
        antialiased=args.antialiased,
        filter_size=args.filter_size,
        replicate_padding=args.replicate_padding,
        use_normalization=not args.no_normalization,
        batch_size=args.batch_size,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        opt=args.opt,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        lr_min=args.lr_min,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_warmup_method=args.lr_warmup_method,
        lr_warmup_decay=args.lr_warmup_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        random_erase=args.random_erase,
        auto_augment=args.auto_augment,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        clip_grad_norm=args.clip_grad_norm,
        tf32=not args.no_tf32,
        matmul_precision=args.matmul_precision,
        compile=args.compile,
        compile_mode=args.compile_mode,
        model_ema=args.model_ema,
        model_ema_steps=args.model_ema_steps,
        model_ema_decay=args.model_ema_decay,
        device=args.device,
        output_dir=args.output_dir,
        resume=args.resume,
        test_only=args.test_only,
        skip_val=args.skip_val,
        print_freq=args.print_freq,
        save_freq=args.save_freq,
        deterministic=args.deterministic,
        sync_bn=args.sync_bn,
    )

    return config


if __name__ == "__main__":
    # Check for extract subcommand
    if len(sys.argv) >= 2 and sys.argv[1] == "extract":
        if len(sys.argv) != 4:
            print(
                "Usage: python train_vgg19.py extract <checkpoint.pth> <output_features.pth>"
            )
            sys.exit(1)
        extract_features_weights(sys.argv[2], sys.argv[3])
    else:
        config = parse_args()
        main(config)
