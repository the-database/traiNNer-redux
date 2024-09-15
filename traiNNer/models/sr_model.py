import os
import shutil
from collections import OrderedDict
from os import path as osp
from typing import Any

import torch
from schedulefree import AdamWScheduleFree
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from traiNNer.archs import build_network
from traiNNer.archs.arch_info import ARCHS_WITHOUT_FP16
from traiNNer.data.base_dataset import BaseDataset
from traiNNer.losses import build_loss
from traiNNer.losses.loss_util import get_refined_artifact_map
from traiNNer.metrics import calculate_metric
from traiNNer.models.base_model import BaseModel
from traiNNer.utils import get_root_logger, imwrite, tensor2img
from traiNNer.utils.options import struct2dict
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed


class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)

        # define network
        self.net_g = build_network({**opt.network_g, "scale": opt.scale})
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        if self.opt.path.pretrain_network_g is not None:
            self.load_network(
                self.net_g,
                self.opt.path.pretrain_network_g,
                self.opt.path.strict_load_g,
                self.opt.path.param_key_g,
            )

        self.lq: Tensor | None = None
        self.gt: Tensor | None = None
        self.output: Tensor | None = None
        logger = get_root_logger()

        # use amp
        self.use_amp = self.opt.use_amp
        self.amp_dtype = torch.bfloat16 if self.opt.amp_bf16 else torch.float16

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

        if self.opt.fast_matmul:
            logger.info(
                "Fast matrix multiplication and convolution operations (fast_matmul) enabled, trading precision for performance."
            )

        if self.is_train and self.opt.train:
            # define network net_d if GAN is enabled
            self.has_gan = False
            gan_opt = self.opt.train.gan_opt
            if gan_opt:
                if gan_opt.get("loss_weight", 0) > 0:
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
                    self.net_d = self.model_to_device(self.net_d)
                    # self.print_network(self.net_d)

                    # load pretrained models
                    if self.opt.path.pretrain_network_d is not None:
                        self.load_network(
                            self.net_d,
                            self.opt.path.pretrain_network_d,
                            self.opt.path.strict_load_d,
                            self.opt.path.param_key_d,
                        )

            self.cri_pix = None
            self.cri_mssim = None
            self.cri_mssim_l1 = None
            self.cri_ldl = None
            self.cri_dists = None
            self.cri_perceptual = None
            self.cri_contextual = None
            self.cri_color = None
            self.cri_luma = None
            self.cri_hsluv = None
            self.cri_gan = None
            self.cri_avg = None
            self.cri_bicubic = None

            self.ema_decay = 0
            self.net_g_ema = None

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

        self.scaler_g = GradScaler(enabled=self.use_amp, device="cuda")
        self.scaler_d = GradScaler(enabled=self.use_amp, device="cuda")

        self.accum_iters = self.opt.datasets["train"].accum_iter

        self.ema_decay = train_opt.ema_decay
        if self.ema_decay > 0:
            logger.info(
                "Using Exponential Moving Average (EMA) with decay: %s.", self.ema_decay
            )

            # load pretrained model
            init_net_g_ema = build_network(
                {**self.opt.network_g, "scale": self.opt.scale}
            ).to(self.device)
            if self.opt.path.pretrain_network_g_ema is not None:
                self.load_network(
                    init_net_g_ema,
                    self.opt.path.pretrain_network_g_ema,
                    self.opt.path.strict_load_g,
                    "params_ema",
                )

            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = AveragedModel(
                init_net_g_ema,
                multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay),
                device=self.device,
            )

        self.grad_clip = train_opt.grad_clip
        if self.grad_clip:
            logger.info("Gradient clipping is enabled.")

        # define losses
        if train_opt.pixel_opt:
            if train_opt.pixel_opt.get("loss_weight", 0) > 0:
                self.cri_pix = build_loss(train_opt.pixel_opt).to(self.device)

        if train_opt.mssim_opt:
            if train_opt.mssim_opt.get("loss_weight", 0) > 0:
                self.cri_mssim = build_loss(train_opt.mssim_opt).to(self.device)

        if train_opt.mssim_l1_opt:
            if train_opt.mssim_l1_opt.get("loss_weight", 0) > 0:
                self.cri_mssim_l1 = build_loss(train_opt.mssim_l1_opt).to(self.device)

        if train_opt.ldl_opt:
            if train_opt.ldl_opt.get("loss_weight", 0) > 0:
                self.cri_ldl = build_loss(train_opt.ldl_opt).to(self.device)

        if train_opt.perceptual_opt:
            if (
                train_opt.perceptual_opt.get("perceptual_weight", 0) > 0
                or train_opt.perceptual_opt.get("style_weight", 0) > 0
            ):
                self.cri_perceptual = build_loss(train_opt.perceptual_opt).to(
                    self.device
                )

        if train_opt.dists_opt:
            if train_opt.dists_opt.get("loss_weight", 0) > 0:
                self.cri_dists = build_loss(train_opt.dists_opt).to(self.device)

        if train_opt.contextual_opt:
            if train_opt.contextual_opt.get("loss_weight", 0) > 0:
                self.cri_contextual = build_loss(train_opt.contextual_opt).to(
                    self.device
                )

        if train_opt.color_opt:
            if train_opt.color_opt.get("loss_weight", 0) > 0:
                self.cri_color = build_loss(train_opt.color_opt).to(self.device)

        if train_opt.luma_opt:
            if train_opt.luma_opt.get("loss_weight", 0) > 0:
                self.cri_luma = build_loss(train_opt.luma_opt).to(self.device)

        if train_opt.hsluv_opt:
            if train_opt.hsluv_opt.get("loss_weight", 0) > 0:
                self.cri_hsluv = build_loss(train_opt.hsluv_opt).to(self.device)

        if train_opt.avg_opt:
            if train_opt.avg_opt.get("loss_weight", 0) > 0:
                self.cri_avg = build_loss(train_opt.avg_opt).to(self.device)

        if train_opt.bicubic_opt:
            if train_opt.bicubic_opt.get("loss_weight", 0) > 0:
                self.cri_bicubic = build_loss(train_opt.bicubic_opt).to(self.device)

        if train_opt.gan_opt:
            if train_opt.gan_opt.get("loss_weight", 0) > 0:
                self.cri_gan = build_loss(train_opt.gan_opt).to(self.device)
        else:
            self.cri_gan = None

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
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning("Params %s will not be optimized.", k)

        optim_g_opts = struct2dict(train_opt.optim_g)
        optim_type = optim_g_opts.pop("type")
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **optim_g_opts)
        self.optimizers.append(self.optimizer_g)
        self.optimizers_skipped.append(False)
        self.optimizers_schedule_free.append("ScheduleFree" in optim_type)

        # optimizer d
        if self.net_d is not None:
            assert train_opt.optim_d is not None
            optim_d_opts = struct2dict(train_opt.optim_d)
            optim_type = optim_d_opts.pop("type")
            self.optimizer_d = self.get_optimizer(
                optim_type, self.net_d.parameters(), **optim_d_opts
            )
            self.optimizers.append(self.optimizer_d)
            self.optimizers_skipped.append(False)
            self.optimizers_schedule_free.append("ScheduleFree" in optim_type)

    def feed_data(self, data: DataFeed) -> None:
        assert "lq" in data
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

        # moa
        if self.is_train and self.batch_augment and self.gt is not None:
            self.gt, self.lq = self.batch_augment(self.gt, self.lq)

    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        # https://github.com/Corpsecreate/neosr/blob/2ee3e7fe5ce485e070744158d4e31b8419103db0/neosr/models/default.py#L328

        assert self.optimizer_g is not None
        assert self.lq is not None
        assert self.gt is not None
        assert self.scaler_d is not None
        assert self.scaler_g is not None

        # optimize net_d
        if self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False

        n_samples = self.gt.shape[0]
        self.loss_samples += n_samples

        # TODO refactor self.accum_iters
        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            self.output = self.net_g(self.lq)
            assert isinstance(self.output, Tensor)
            l_g_total = torch.tensor(0.0, device=self.output.device)
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt) / self.accum_iters
                l_g_total += l_g_pix
                loss_dict["l_g_pix"] = l_g_pix
            if self.cri_mssim:
                l_g_mssim = self.cri_mssim(self.output, self.gt) / self.accum_iters
                l_g_total += l_g_mssim
                loss_dict["l_g_mssim"] = l_g_mssim
            if self.cri_mssim_l1:
                l_g_mssim_l1 = (
                    self.cri_mssim_l1(self.output, self.gt) / self.accum_iters
                )
                l_g_total += l_g_mssim_l1
                loss_dict["l_g_mssim_l1"] = l_g_mssim_l1
            if self.cri_ldl:
                assert self.net_g_ema is not None
                # TODO support LDL without ema
                pixel_weight = get_refined_artifact_map(
                    self.gt, self.output, self.net_g_ema(self.lq), 7
                )
                l_g_ldl = (
                    self.cri_ldl(
                        torch.mul(pixel_weight, self.output),
                        torch.mul(pixel_weight, self.gt),
                    )
                    / self.accum_iters
                )
                l_g_total += l_g_ldl
                loss_dict["l_g_ldl"] = l_g_ldl
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_percep /= self.accum_iters
                    l_g_total += l_g_percep
                    loss_dict["l_g_percep"] = l_g_percep
                if l_g_style is not None:
                    l_g_total /= self.accum_iters
                    l_g_total += l_g_style
                    loss_dict["l_g_style"] = l_g_style
            # dists loss
            if self.cri_dists:
                l_g_dists = self.cri_dists(self.output, self.gt) / self.accum_iters
                l_g_total += l_g_dists
                loss_dict["l_g_dists"] = l_g_dists
            # contextual loss
            if self.cri_contextual:
                l_g_contextual = (
                    self.cri_contextual(self.output, self.gt) / self.accum_iters
                )
                l_g_total += l_g_contextual
                loss_dict["l_g_contextual"] = l_g_contextual
            # color loss
            if self.cri_color:
                l_g_color = self.cri_color(self.output, self.gt) / self.accum_iters
                l_g_total += l_g_color
                loss_dict["l_g_color"] = l_g_color
            # luma loss
            if self.cri_luma:
                l_g_luma = self.cri_luma(self.output, self.gt) / self.accum_iters
                l_g_total += l_g_luma
                loss_dict["l_g_luma"] = l_g_luma
            # hsluv loss
            if self.cri_hsluv:
                l_g_hsluv = self.cri_hsluv(self.output, self.gt) / self.accum_iters
                l_g_total += l_g_hsluv
                loss_dict["l_g_hsluv"] = l_g_hsluv
            # avg loss
            if self.cri_avg:
                l_g_avg = self.cri_avg(self.output, self.gt) / self.accum_iters
                l_g_total += l_g_avg
                loss_dict["l_g_avg"] = l_g_avg
            # bicubic loss
            if self.cri_bicubic:
                l_g_bicubic = self.cri_bicubic(self.output, self.gt) / self.accum_iters
                l_g_total += l_g_bicubic
                loss_dict["l_g_bicubic"] = l_g_bicubic
            # gan loss
            if self.cri_gan and self.net_d:
                fake_g_pred = self.net_d(self.output)
                l_g_gan = (
                    self.cri_gan(fake_g_pred, True, is_disc=False) / self.accum_iters
                )
                l_g_total += l_g_gan
                loss_dict["l_g_gan"] = l_g_gan

            # add total generator loss for tensorboard tracking
            loss_dict["l_g_total"] = l_g_total

        self.scaler_g.scale(l_g_total).backward()
        if apply_gradient:
            if self.grad_clip:
                self.scaler_g.unscale_(self.optimizer_g)
                clip_grad_norm_(self.net_g.parameters(), 1.0)

            scale_before = self.scaler_g.get_scale()
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
            self.optimizers_skipped[0] = self.scaler_g.get_scale() < scale_before
            self.optimizer_g.zero_grad()

        if (
            self.net_d is not None
            and self.cri_gan is not None
            and self.optimizer_d is not None
        ):
            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                # real
                real_d_pred = self.net_d(self.gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                loss_dict["l_d_real"] = l_d_real
                loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())
                # fake
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                loss_dict["l_d_fake"] = l_d_fake
                loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())

            self.scaler_d.scale(l_d_real).backward()
            self.scaler_d.scale(l_d_fake).backward()
            if apply_gradient:
                if self.grad_clip:
                    self.scaler_d.unscale_(self.optimizer_d)
                    clip_grad_norm_(self.net_d.parameters(), 1.0)
                scale_before = self.scaler_d.get_scale()
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
                self.optimizers_skipped[1] = self.scaler_d.get_scale() < scale_before
                self.optimizer_d.zero_grad()

        for key, value in loss_dict.items():
            val = (
                value
                if isinstance(value, float)
                else value.to(dtype=torch.float32).detach()
            )
            self.log_dict[key] = self.log_dict.get(key, 0) + val * n_samples

        if self.net_g_ema is not None and apply_gradient:
            if not (self.use_amp and self.optimizers_skipped[0]):
                self.net_g_ema.update_parameters(self.net_g)

    def test(self) -> None:
        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            if self.optimizers_schedule_free[0]:
                assert isinstance(self.optimizer_g, AdamWScheduleFree)
                self.optimizer_g.eval()

            if self.net_g_ema is not None:
                self.net_g_ema.eval()
                with torch.inference_mode():
                    self.output = self.net_g_ema(self.lq)
            else:
                self.net_g.eval()
                with torch.inference_mode():
                    self.output = self.net_g(self.lq)
                self.net_g.train()

            if self.optimizers_schedule_free[0]:
                assert isinstance(self.optimizer_g, AdamWScheduleFree)
                self.optimizer_g.train()

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
                self.metric_results: dict[str, Any] = {
                    metric: 0 for metric in self.opt.val.metrics.keys()
                }
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if self.with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = {}
        pbar = None
        if self.use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")

        logger = get_root_logger()
        if save_img and len(dataloader) > 0:
            logger.info(
                "Saving %d validation images to: %s",
                len(dataloader),
                self.opt.path.visualization,
            )

        for val_data in dataloader:
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img(visuals["result"])
            metric_data["img"] = sr_img
            if "gt" in visuals:
                gt_img = tensor2img(visuals["gt"])
                metric_data["img2"] = gt_img
                self.gt = None

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
                        save_img_dir = osp.join(
                            self.opt.path.visualization,
                            osp.relpath(
                                osp.splitext(val_data["lq_path"][0])[0],
                                dataloader.dataset.opt.dataroot_lq,
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
                        f"{img_name}_{self.opt.name}.png",
                    )
                imwrite(sr_img, save_img_path)
                if (
                    self.opt.is_train
                    and not self.first_val_completed
                    and "lq_path" in val_data
                ):
                    assert save_img_dir is not None
                    lr_img_target_path = osp.join(save_img_dir, f"{img_name}_lr.png")
                    if not os.path.exists(lr_img_target_path):
                        shutil.copy(val_data["lq_path"][0], lr_img_target_path)

            if self.with_metrics:
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

        if self.with_metrics:
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
            log_str += f"\t # {metric}: {value:.4f}"
            if len(self.best_metric_results) > 0:
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
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
        current_accum_iter: int,
        apply_gradient: bool,
    ) -> None:
        assert self.opt.path.models is not None
        assert self.opt.path.resume_models is not None

        if self.net_g_ema is not None:
            self.save_network(
                self.net_g_ema,
                "net_g_ema",
                self.opt.path.models,
                current_iter,
                current_accum_iter,
                apply_gradient,
            )

            self.save_network(
                self.net_g,
                "net_g",
                self.opt.path.resume_models,
                current_iter,
                current_accum_iter,
                apply_gradient,
            )
        else:
            self.save_network(
                self.net_g,
                "net_g",
                self.opt.path.models,
                current_iter,
                current_accum_iter,
                apply_gradient,
            )

        if self.net_d is not None:
            self.save_network(
                self.net_d,
                "net_d",
                self.opt.path.resume_models,
                current_iter,
                current_accum_iter,
                apply_gradient,
            )

        self.save_training_state(
            epoch, current_iter, current_accum_iter, apply_gradient
        )
