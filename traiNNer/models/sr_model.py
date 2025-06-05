import os
import shutil
import warnings
from collections import OrderedDict
from os import path as osp
from typing import Any

import cv2
import torch
from ema_pytorch import EMA
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional as F  # noqa: N812
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from traiNNer.archs import build_network
from traiNNer.archs.arch_info import ARCHS_WITHOUT_FP16
from traiNNer.data.base_dataset import BaseDataset
from traiNNer.losses import build_loss
from traiNNer.metrics import calculate_metric
from traiNNer.models.base_model import BaseModel
from traiNNer.utils import get_root_logger, imwrite, tensor2img
from traiNNer.utils.color_util import pixelformat2rgb_pt, rgb2pixelformat_pt
from traiNNer.utils.logger import clickable_file_path
from traiNNer.utils.misc import loss_type_to_label
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed


class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)

        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

        # use amp
        self.use_amp = self.opt.use_amp
        self.use_channels_last = self.opt.use_channels_last
        self.memory_format = (
            torch.channels_last
            if self.use_amp and self.use_channels_last
            else torch.preserve_format
        )
        self.amp_dtype = torch.bfloat16 if self.opt.amp_bf16 else torch.float16
        self.use_compile = self.opt.use_compile

        # define network
        assert opt.network_g is not None, "network_g must be defined"
        self.net_g = build_network({**opt.network_g, "scale": opt.scale})

        # load pretrained models
        if self.opt.path.pretrain_network_g is not None:
            self.load_network(
                self.net_g,
                self.opt.path.pretrain_network_g,
                self.opt.path.strict_load_g,
                self.opt.path.param_key_g,
            )

        self.net_g = self.model_to_device(self.net_g)

        self.lq: Tensor | None = None
        self.gt: Tensor | None = None
        self.output: Tensor | None = None
        logger = get_root_logger()

        if self.use_amp:
            if self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                logger.warning(
                    "bf16 was enabled for AMP but the current GPU does not support bf16. Falling back to float16 for AMP. Disable bf16 to hide this warning (amp_bf16: false)."
                )
                self.amp_dtype = torch.float16

            network_g_name = opt.network_g["type"]
            if (
                self.amp_dtype == torch.float16
                and network_g_name.lower() in ARCHS_WITHOUT_FP16
            ):
                if torch.cuda.is_bf16_supported():
                    logger.warning(
                        "AMP with fp16 was enabled but network_g [bold]%s[/bold] does not support fp16. Falling back to bf16.",
                        network_g_name,
                        extra={"markup": True},
                    )
                    self.amp_dtype = torch.bfloat16
                else:
                    logger.warning(
                        "AMP with fp16 was enabled but network_g [bold]%s[/bold] does not support fp16. Disabling AMP.",
                        network_g_name,
                        extra={"markup": True},
                    )
                    self.use_amp = False
        elif self.amp_dtype == torch.bfloat16:
            logger.warning(
                "bf16 was enabled without AMP and will have no effect. Enable AMP to use bf16 (use_amp: true)."
            )

        if self.use_amp:
            logger.info(
                "Using Automatic Mixed Precision (AMP) with fp32 and %s.",
                "bf16" if self.amp_dtype == torch.bfloat16 else "fp16",
            )

            if self.use_channels_last:
                logger.info("Using channels last memory format.")

        if self.opt.fast_matmul:
            logger.info(
                "Fast matrix multiplication and convolution operations (fast_matmul) enabled, trading precision for performance."
            )

        if self.is_train and self.opt.train:
            # define network net_d if GAN is enabled
            self.has_gan = False
            gan_opt = self.opt.train.gan_opt

            if not gan_opt:
                if self.opt.train.losses:
                    gan_opts = list(
                        filter(
                            lambda x: x["type"].lower() == "ganloss",
                            self.opt.train.losses,
                        )
                    )
                    if gan_opts:
                        gan_opt = gan_opts[0]

            if gan_opt:
                if gan_opt.get("loss_weight", 0) != 0:
                    self.has_gan = True

            self.net_d = None
            if self.has_gan:
                if self.opt.train.optim_d is None:
                    raise ValueError(
                        "GAN loss requires discriminator optimizer (optim_d). Define optim_d or disable GAN loss."
                    )
                if self.opt.network_d is None:
                    raise ValueError(
                        "GAN loss requires discriminator network (network_d). Define network_d or disable GAN loss."
                    )
                else:
                    self.net_d = build_network(self.opt.network_d)
                    # load pretrained models
                    if self.opt.path.pretrain_network_d is not None:
                        self.load_network(
                            self.net_d,
                            self.opt.path.pretrain_network_d,
                            self.opt.path.strict_load_d,
                            self.opt.path.param_key_d,
                        )
                    self.net_d = self.model_to_device(self.net_d)

            self.losses = {}

            self.ema_decay = 0
            self.net_g_ema: EMA | None = None

            self.optimizer_g: Optimizer | None = None
            self.optimizer_d: Optimizer | None = None

            self.init_training_settings()

    def init_training_settings(self) -> None:
        self.net_g.train()
        if self.net_d is not None:
            self.net_d.train()

        train_opt = self.opt.train
        assert train_opt is not None

        logger = get_root_logger()

        enable_gradscaler = self.use_amp and not self.opt.amp_bf16

        self.scaler_g = GradScaler(enabled=enable_gradscaler, device="cuda")
        self.scaler_d = GradScaler(enabled=enable_gradscaler, device="cuda")

        self.accum_iters = self.opt.datasets["train"].accum_iter

        self.adaptive_d = train_opt.adaptive_d
        self.adaptive_d_ema_decay = train_opt.adaptive_d_ema_decay
        self.adaptive_d_threshold = train_opt.adaptive_d_threshold
        self.ema_decay = train_opt.ema_decay

        if self.ema_decay > 0:
            logger.info(
                "Using Exponential Moving Average (EMA) with decay: %s.", self.ema_decay
            )
            assert self.opt.network_g is not None, "network_g must be defined"

            init_net_g_ema = None

            # load pretrained model
            if self.opt.path.pretrain_network_g_ema is not None:
                init_net_g_ema = build_network(
                    {**self.opt.network_g, "scale": self.opt.scale}
                )
                self.load_network(
                    init_net_g_ema,
                    self.opt.path.pretrain_network_g_ema,
                    self.opt.path.strict_load_g,
                    "params_ema",
                )

            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            switch_iter = train_opt.ema_switch_iter
            if switch_iter == 0:
                switch_iter = None
            self.net_g_ema = EMA(
                self.get_bare_model(self.net_g),
                ema_model=init_net_g_ema,
                beta=self.ema_decay,
                allow_different_devices=True,
                update_after_step=train_opt.ema_update_after_step,
                update_every=1,
                power=train_opt.ema_power,
                update_model_with_ema_every=switch_iter,
            ).to(device=self.device, memory_format=self.memory_format)  # pyright: ignore[reportCallIssue]

            assert self.net_g_ema is not None
            self.net_g_ema.step = self.net_g_ema.step.to(device=torch.device("cpu"))

        self.grad_clip = train_opt.grad_clip
        if self.grad_clip:
            logger.info("Gradient clipping is enabled.")

        # define losses

        if train_opt.losses is None:
            train_opt.losses = []
            # old loss format
            old_loss_opts = [
                "pixel_opt",
                "mssim_opt",
                "ms_ssim_l1_opt",
                "perceptual_opt",
                "contextual_opt",
                "dists_opt",
                "hr_inversion_opt",
                "dinov2_opt",
                "topiq_opt",
                "pd_opt",
                "fd_opt",
                "ldl_opt",
                "hsluv_opt",
                "gan_opt",
                "color_opt",
                "luma_opt",
                "avg_opt",
                "bicubic_opt",
            ]
            for opt in old_loss_opts:
                loss = getattr(train_opt, opt)
                if loss is not None:
                    train_opt.losses.append(loss)

        for loss in train_opt.losses:
            assert "type" in loss, "all losses must define type"
            assert "loss_weight" in loss, f"{loss['type']} must define loss_weight"
            if float(loss["loss_weight"]) != 0:
                label = loss_type_to_label(loss["type"])
                if label == "l_g_gan":
                    self.has_gan = True
                self.losses[label] = build_loss(loss).to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )  # pyright: ignore[reportCallIssue] # https://github.com/pytorch/pytorch/issues/131765

                if self.use_compile:
                    logger.info(
                        "Compiling loss %s. This may take several minutes...", label
                    )
                    self.losses[label] = torch.compile(self.losses[label])

        assert self.losses, "At least one loss must be defined."

        if not self.has_gan:
            # warn that discriminator network / optimizer won't be used if enabled
            if self.opt.network_d is not None:
                logger.warning(
                    "Discriminator network (network_d) is defined but GAN loss is disabled. Discriminator network will have no effect."
                )

            if train_opt.optim_d is not None:
                logger.warning(
                    "Discriminator optimizer (optim_d) is defined but GAN loss is disabled. Discriminator optimizer will have no effect."
                )

        # setup batch augmentations
        self.setup_batchaug()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self) -> None:
        train_opt = self.opt.train
        assert train_opt is not None
        # assert train_opt.optim_g is not None
        optim_params = []
        logger = get_root_logger()

        if train_opt.optim_g is not None:
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                elif "eval_" in k:
                    pass  # intentionally frozen for reparameterization, skip warning
                else:
                    logger.warning("Params %s will not be optimized.", k)

            self.optimizer_g = self.get_optimizer(optim_params, train_opt.optim_g)
            self.optimizers.append(self.optimizer_g)
            self.optimizers_skipped.append(False)
            self.optimizers_schedule_free.append(
                "SCHEDULEFREE" in train_opt.optim_g["type"].upper()
            )
        else:
            logger.warning("!!! net_g will not be optimized. !!!")

        # optimizer d
        if self.net_d is not None:
            assert train_opt.optim_d is not None
            self.optimizer_d = self.get_optimizer(
                self.net_d.parameters(), train_opt.optim_d
            )
            self.optimizers.append(self.optimizer_d)
            self.optimizers_skipped.append(False)
            self.optimizers_schedule_free.append(
                "SCHEDULEFREE" in train_opt.optim_d["type"].upper()
            )

    def feed_data(self, data: DataFeed) -> None:
        assert "lq" in data
        self.lq = data["lq"].to(
            self.device,
            memory_format=self.memory_format,
            non_blocking=True,
        )
        if "gt" in data:
            self.gt = data["gt"].to(
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )

        # moa
        if self.is_train and self.batch_augment and self.gt is not None:
            self.gt, self.lq = self.batch_augment(self.gt, self.lq)

    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        # https://github.com/Corpsecreate/neosr/blob/2ee3e7fe5ce485e070744158d4e31b8419103db0/neosr/models/default.py#L328
        # assert self.optimizer_g is not None
        assert self.lq is not None
        assert self.gt is not None
        assert self.scaler_d is not None
        assert self.scaler_g is not None

        skip_d_update = False

        # optimize net_d
        if self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False

        n_samples = self.gt.shape[0]
        self.loss_samples += n_samples
        loss_dict: dict[str, Tensor | float] = OrderedDict()

        lq = rgb2pixelformat_pt(
            self.lq, self.opt.input_pixel_format
        )  # lq: input_pixel_format
        rgb2pixelformat_pt(
            self.gt, self.opt.input_pixel_format
        )  # gt: input_pixel_format

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            if self.optimizer_g is not None:
                output = self.net_g(lq)  # output: output_pixel_format
                self.output = pixelformat2rgb_pt(
                    output, self.gt, self.opt.output_pixel_format
                )  # self.output: rgb

                assert isinstance(self.output, Tensor)
                l_g_total = torch.tensor(0.0, device=self.output.device)

                lq_target = None

                for label, loss in self.losses.items():
                    target = self.gt

                    if loss.loss_weight < 0:
                        if lq_target is None:
                            with torch.inference_mode():
                                lq_target = torch.clamp(
                                    F.interpolate(
                                        self.lq,
                                        scale_factor=self.opt.scale,
                                        mode="bicubic",
                                        antialias=True,
                                    ),
                                    0,
                                    1,
                                )
                        target = lq_target

                    if label == "l_g_gan":
                        assert self.net_d is not None
                        fake_g_pred = self.net_d(self.output)
                        l_g_loss = loss(fake_g_pred, True, is_disc=False)

                        if self.adaptive_d:
                            l_g_gan_ema = (
                                self.adaptive_d_ema_decay * self.l_g_gan_ema
                                + (1 - self.adaptive_d_ema_decay) * l_g_loss.detach()
                            )

                            if (
                                l_g_gan_ema
                                > self.l_g_gan_ema * self.adaptive_d_threshold
                            ):
                                skip_d_update = True
                                self.optimizers_skipped[1] = True
                                # print(current_iter, "skip_d_update")

                            self.l_g_gan_ema = l_g_gan_ema

                    elif label == "l_g_ldl":
                        assert self.net_g_ema is not None, (
                            "ema_decay must be enabled for LDL loss"
                        )
                        with torch.inference_mode():
                            output_ema = pixelformat2rgb_pt(
                                self.net_g_ema(lq),
                                self.gt,
                                self.opt.output_pixel_format,
                            )
                        l_g_loss = loss(self.output, output_ema, target)
                    else:
                        l_g_loss = loss(self.output, target)

                    if isinstance(l_g_loss, dict):
                        for sublabel, loss_val in l_g_loss.items():
                            if loss_val > 0:
                                weighted_loss_val = loss_val * abs(loss.loss_weight)
                                l_g_total += weighted_loss_val * self.accum_iters
                                loss_dict[f"{label}_{sublabel}"] = weighted_loss_val
                    else:
                        weighted_l_g_loss = l_g_loss * abs(loss.loss_weight)
                        l_g_total += weighted_l_g_loss / self.accum_iters
                        loss_dict[label] = weighted_l_g_loss

                if not l_g_total.isfinite():
                    raise RuntimeError(
                        "Training failed: NaN/Inf found in loss. Try reducing the learning rate. If training still fails, please file an issue: https://github.com/the-database/traiNNer-redux/issues"
                    )

                # add total generator loss for tensorboard tracking
                loss_dict["l_g_total"] = l_g_total

                self.scaler_g.scale(l_g_total).backward()

                if apply_gradient:
                    self.scaler_g.unscale_(self.optimizer_g)
                    grad_norm_g = torch.linalg.vector_norm(
                        torch.stack(
                            [
                                torch.linalg.vector_norm(p.grad, 2)
                                for p in self.net_g.parameters()
                                if p.grad is not None
                            ]
                        )
                    ).detach()

                    loss_dict["grad_norm_g"] = grad_norm_g

                    if self.grad_clip:
                        clip_grad_norm_(self.net_g.parameters(), 1.0)

                    scale_before = self.scaler_g.get_scale()
                    self.scaler_g.step(self.optimizer_g)
                    self.scaler_g.update()
                    scale_after = self.scaler_g.get_scale()
                    loss_dict["scale_g"] = scale_after
                    self.optimizers_skipped[0] = scale_after < scale_before
                    self.optimizer_g.zero_grad()
            else:
                with torch.inference_mode():
                    self.output = self.net_g(self.lq)
                    assert isinstance(self.output, Tensor)

        cri_gan = self.losses.get("l_g_gan")

        if (
            self.net_d is not None
            and cri_gan is not None
            and self.optimizer_d is not None
            and not skip_d_update
        ):
            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                # real
                real_d_pred = self.net_d(self.gt)
                l_d_real = cri_gan(real_d_pred, True, is_disc=True)
                loss_dict["l_d_real"] = l_d_real
                loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())
                # fake
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = cri_gan(fake_d_pred, False, is_disc=True)
                loss_dict["l_d_fake"] = l_d_fake
                loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())

            self.scaler_d.scale((l_d_real + l_d_fake) / self.accum_iters).backward()

            if apply_gradient:
                self.scaler_d.unscale_(self.optimizer_d)
                grad_norm_d = torch.linalg.vector_norm(
                    torch.stack(
                        [
                            torch.linalg.vector_norm(p.grad, 2)
                            for p in self.net_d.parameters()
                            if p.grad is not None
                        ]
                    )
                ).detach()

                loss_dict["grad_norm_d"] = grad_norm_d

                if self.grad_clip:
                    clip_grad_norm_(self.net_d.parameters(), 1.0)
                scale_before = self.scaler_d.get_scale()
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
                scale_after = self.scaler_d.get_scale()
                loss_dict["scale_d"] = scale_after
                self.optimizers_skipped[-1] = scale_after < scale_before
                self.optimizer_d.zero_grad()

        for key, value in loss_dict.items():
            val = (
                value
                if isinstance(value, float)
                else value.to(dtype=torch.float32).detach()  # pyright: ignore[reportAttributeAccessIssue]
            )
            self.log_dict[key] = self.log_dict.get(key, 0) + val * n_samples

        self.log_dict = self.reduce_loss_dict(self.log_dict)

        if self.net_g_ema is not None and apply_gradient:
            if not (self.use_amp and self.optimizers_skipped[0]):
                self.net_g_ema.update()

    def infer_tiled(self, net: nn.Module, lq: torch.Tensor) -> torch.Tensor:
        assert self.opt.val is not None
        tile_size = self.opt.val.tile_size
        tile_overlap = self.opt.val.tile_overlap
        scale = self.opt.scale

        b, c, h, w = lq.shape
        assert b == 1, "Only batch size 1 is supported for tiled inference"

        if h <= tile_size and w <= tile_size:
            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                return net(lq)

        pad_h = (tile_size - (h % tile_size)) % tile_size if h > tile_size else 0
        pad_w = (tile_size - (w % tile_size)) % tile_size if w > tile_size else 0

        lq = torch.nn.functional.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, h_pad, w_pad = lq.shape

        output = torch.zeros((1, c, h_pad * scale, w_pad * scale), device=lq.device)
        weight_map = torch.zeros_like(output)

        hr_tile = tile_size * scale
        wy = torch.linspace(0, 1, hr_tile, device=lq.device)
        wx = torch.linspace(0, 1, hr_tile, device=lq.device)
        wy = 1 - torch.abs(wy - 0.5) * 2
        wx = 1 - torch.abs(wx - 0.5) * 2
        weight = torch.ger(wy, wx).unsqueeze(0).unsqueeze(0)

        stride = tile_size - tile_overlap
        tiles_y = max(1, (h_pad - tile_overlap + stride - 1) // stride)
        tiles_x = max(1, (w_pad - tile_overlap + stride - 1) // stride)

        for y in range(tiles_y):
            for x in range(tiles_x):
                in_y0 = y * stride
                in_x0 = x * stride
                in_y1 = min(in_y0 + tile_size, h_pad)
                in_x1 = min(in_x0 + tile_size, w_pad)

                lq_patch = lq[:, :, in_y0:in_y1, in_x0:in_x1]

                ph, pw = lq_patch.shape[-2:]
                pad_bottom = max(tile_size - ph, 0) if ph < tile_size else 0
                pad_right = max(tile_size - pw, 0) if pw < tile_size else 0

                if pad_bottom > 0 or pad_right > 0:
                    pad_bottom = min(pad_bottom, ph - 1)
                    pad_right = min(pad_right, pw - 1)
                    lq_patch = torch.nn.functional.pad(
                        lq_patch,
                        (0, pad_right, 0, pad_bottom),
                        mode="reflect",
                    )

                out_patch = net(lq_patch)
                out_patch = out_patch[:, :, : ph * scale, : pw * scale]
                w_patch = weight[:, :, : ph * scale, : pw * scale]

                out_y0 = in_y0 * scale
                out_x0 = in_x0 * scale
                out_y1 = out_y0 + ph * scale
                out_x1 = out_x0 + pw * scale

                output[:, :, out_y0:out_y1, out_x0:out_x1] += out_patch * w_patch
                weight_map[:, :, out_y0:out_y1, out_x0:out_x1] += w_patch

        out_final = output / weight_map.clamp(min=1e-6)
        return out_final[:, :, : h * scale, : w * scale]

    def test(self) -> None:
        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            if self.optimizers_schedule_free and self.optimizers_schedule_free[0]:
                assert self.optimizer_g is not None
                self.optimizer_g.eval()  # pyright: ignore[reportAttributeAccessIssue]

            assert self.lq is not None

            lq = rgb2pixelformat_pt(
                self.lq, self.opt.input_pixel_format
            )  # lq: input_pixel_format

            net = self.net_g_ema if self.net_g_ema is not None else self.net_g
            net.eval()

            assert self.opt.val is not None
            with torch.inference_mode():
                if self.opt.val.tile_size > 0:
                    tmp_out = self.infer_tiled(net, lq)
                else:
                    tmp_out = net(lq)
                self.output = pixelformat2rgb_pt(
                    tmp_out, self.gt, self.opt.output_pixel_format
                )

            if self.net_g_ema is None:
                net.train()

            if self.optimizers_schedule_free and self.optimizers_schedule_free[0]:
                assert self.optimizer_g is not None
                self.optimizer_g.train()  # pyright: ignore[reportAttributeAccessIssue]

    def dist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        if self.opt.rank == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, multi_val_datasets
            )

    def nondist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        self.is_train = False

        assert isinstance(dataloader.dataset, BaseDataset)
        assert self.opt.val is not None
        assert self.opt.path.visualization is not None

        dataset_name = dataloader.dataset.opt.name

        if self.with_metrics:
            assert self.opt.val.metrics is not None
            if len(self.metric_results) == 0:  # only execute in the first run
                self.metric_results: dict[str, Any] = dict.fromkeys(
                    self.opt.val.metrics.keys(), 0
                )
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if self.with_metrics:
            self.metric_results = dict.fromkeys(self.metric_results, 0)

        metric_data = {}
        pbar = None
        if self.use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")

        logger = get_root_logger()
        if save_img and len(dataloader) > 0:
            logger.info(
                "Saving %d validation images to %s.",
                len(dataloader),
                clickable_file_path(
                    self.opt.path.visualization, "visualization folder"
                ),
            )

        gt_key = "img2"
        run_metrics = self.with_metrics

        for val_data in dataloader:
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img(
                visuals["result"],
                to_bgr=False,
            )
            metric_data["img"] = sr_img
            if "gt" in visuals:
                gt_img = tensor2img(
                    visuals["gt"],
                    to_bgr=False,
                )
                metric_data[gt_key] = gt_img
                self.gt = None
            else:
                run_metrics = False

            # tentative for out of GPU memory
            self.lq = None
            self.output = None
            torch.cuda.empty_cache()

            save_img_dir = None

            if save_img:
                if self.opt.is_train:
                    if multi_val_datasets:
                        save_img_dir = osp.join(
                            self.opt.path.visualization, f"{dataset_name} - {img_name}"
                        )
                    else:
                        assert dataloader.dataset.opt.dataroot_lq is not None, (
                            "dataroot_lq is required for val set"
                        )
                        lq_path = val_data["lq_path"][0]

                        # multiple root paths are supported, find the correct root path for each lq_path
                        normalized_lq_path = osp.normpath(lq_path)

                        matching_root = None
                        for root in dataloader.dataset.opt.dataroot_lq:
                            normalized_root = osp.normpath(root)
                            if normalized_lq_path.startswith(normalized_root + osp.sep):
                                matching_root = root
                                break

                        if matching_root is None:
                            raise ValueError(
                                f"The lq_path {lq_path} does not match any of the provided dataroot_lq paths."
                            )

                        save_img_dir = osp.join(
                            self.opt.path.visualization,
                            osp.relpath(
                                osp.splitext(lq_path)[0],
                                matching_root,
                            ),
                        )
                    save_img_path = osp.join(
                        save_img_dir, f"{img_name}_{current_iter:06d}.png"
                    )
                elif self.opt.val.suffix:
                    save_img_path = osp.join(
                        self.opt.path.visualization,
                        dataset_name,
                        f"{img_name}_{self.opt.val.suffix}.png",
                    )
                else:
                    save_img_path = osp.join(
                        self.opt.path.visualization,
                        dataset_name,
                        f"{img_name}.png",
                    )
                imwrite(cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR), save_img_path)
                if (
                    self.opt.is_train
                    and not self.first_val_completed
                    and "lq_path" in val_data
                ):
                    assert save_img_dir is not None
                    lr_img_target_path = osp.join(save_img_dir, f"{img_name}_lr.png")
                    if not os.path.exists(lr_img_target_path):
                        shutil.copy(val_data["lq_path"][0], lr_img_target_path)

            if run_metrics:
                # calculate metrics
                assert self.opt.val.metrics is not None
                for name, opt_ in self.opt.val.metrics.items():
                    self.metric_results[name] += calculate_metric(
                        metric_data, opt_, self.device
                    )
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")
        if pbar is not None:
            pbar.close()

        if run_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= len(dataloader)
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        self.first_val_completed = True
        self.is_train = True

    def _log_validation_metric_values(
        self, current_iter: int, dataset_name: str, tb_logger: SummaryWriter | None
    ) -> None:
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric:<5}: {value:7.4f}"
            if len(self.best_metric_results) > 0:
                log_str += (
                    f"\tBest: {self.best_metric_results[dataset_name][metric]['val']:7.4f} @ "
                    f"{self.best_metric_results[dataset_name][metric]['iter']:9,} iter"
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset_name}/{metric}", value, current_iter
                )

    def get_current_visuals(self) -> dict[str, Tensor]:
        assert self.output is not None
        assert self.lq is not None

        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()

        if self.gt is not None:
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(
        self,
        epoch: int,
        current_iter: int,
    ) -> None:
        assert self.opt.path.models is not None
        assert self.opt.path.resume_models is not None

        if self.net_g_ema is not None:
            assert isinstance(self.net_g_ema.ema_model, nn.Module)
            self.save_network(
                self.net_g_ema.ema_model,
                "net_g_ema",
                self.opt.path.models,
                current_iter,
                "params_ema",
            )

            self.save_network(
                self.net_g, "net_g", self.opt.path.resume_models, current_iter, "params"
            )
        else:
            self.save_network(
                self.net_g, "net_g", self.opt.path.models, current_iter, "params"
            )

        if self.net_d is not None:
            self.save_network(
                self.net_d, "net_d", self.opt.path.resume_models, current_iter, "params"
            )

        self.save_training_state(epoch, current_iter)
