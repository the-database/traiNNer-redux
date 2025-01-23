import os
import shutil
import warnings
from collections import OrderedDict
from os import path as osp

import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from traiNNer.archs import build_network
from traiNNer.archs.arch_info import ARCHS_WITHOUT_FP16
from traiNNer.data.base_dataset import BaseDataset
from traiNNer.losses import build_loss
from traiNNer.models.base_model import BaseModel
from traiNNer.utils import get_root_logger, imwrite, tensor2img
from traiNNer.utils.logger import clickable_file_path
from traiNNer.utils.misc import loss_type_to_label
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed


class AEModel(BaseModel):
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
        assert opt.network_ae is not None
        self.net_ae = build_network(
            {**opt.network_ae, "scale": opt.scale, "freeze": False}
        )

        # load pretrained models
        if self.opt.path.pretrain_network_ae is not None:
            self.load_network(
                self.net_ae,
                self.opt.path.pretrain_network_ae,
            )
        elif self.opt.path.pretrain_network_ae_decoder is not None:
            self.load_network(
                self.net_ae.decoder,
                self.opt.path.pretrain_network_ae_decoder,
            )

        self.net_ae = self.model_to_device(self.net_ae)

        self.gt: Tensor | None = None
        self.lq: Tensor | None = None
        self.output_gt: Tensor | None = None
        self.output_lq: Tensor | None = None
        logger = get_root_logger()

        if self.use_amp:
            if self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                logger.warning(
                    "bf16 was enabled for AMP but the current GPU does not support bf16. Falling back to float16 for AMP. Disable bf16 to hide this warning (amp_bf16: false)."
                )
                self.amp_dtype = torch.float16

            network_ae_name = opt.network_ae["type"]
            if (
                self.amp_dtype == torch.float16
                and network_ae_name.lower() in ARCHS_WITHOUT_FP16
            ):
                if torch.cuda.is_bf16_supported():
                    logger.warning(
                        "AMP with fp16 was enabled but network_ae [bold]%s[/bold] does not support fp16. Falling back to bf16.",
                        network_ae_name,
                        extra={"markup": True},
                    )
                    self.amp_dtype = torch.bfloat16
                else:
                    logger.warning(
                        "AMP with fp16 was enabled but network_ae [bold]%s[/bold] does not support fp16. Disabling AMP.",
                        network_ae_name,
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
            self.losses = {}

            self.ema_decay = 0
            self.net_ae_ema = None

            self.optimizer_ae: Optimizer | None = None

            self.init_training_settings()

    def init_training_settings(self) -> None:
        self.net_ae.train()

        train_opt = self.opt.train
        assert train_opt is not None

        logger = get_root_logger()

        self.scaler_ae = GradScaler(enabled=self.use_amp, device="cuda")

        self.accum_iters = self.opt.datasets["train"].accum_iter

        self.ema_decay = train_opt.ema_decay
        if self.ema_decay > 0:
            logger.info(
                "Using Exponential Moving Average (EMA) with decay: %s.", self.ema_decay
            )
            assert self.opt.network_ae is not None
            init_net_ae_ema = build_network(
                {**self.opt.network_ae, "scale": self.opt.scale, "freeze": False}
            )

            # load pretrained model
            if self.opt.path.pretrain_network_ae_ema is not None:
                self.load_network(
                    init_net_ae_ema,
                    self.opt.path.pretrain_network_ae_ema,
                    True,
                    "params_ema",
                )
            elif self.opt.path.pretrain_network_ae_decoder_ema is not None:
                self.load_network(
                    init_net_ae_ema.decoder,
                    self.opt.path.pretrain_network_ae_decoder_ema,
                    True,
                    "params_ema",
                )

            # define network net_ae with Exponential Moving Average (EMA)
            # net_ae_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_ae_ema = AveragedModel(
                init_net_ae_ema.to(memory_format=self.memory_format),  # pyright: ignore[reportCallIssue]
                multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay),
                device=self.device,
            )

            self.net_ae_ema.n_averaged = self.net_ae_ema.n_averaged.to(
                device=torch.device("cpu")
            )

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
            if float(loss["loss_weight"]) > 0:
                label = loss_type_to_label(loss["type"], "ae")
                self.losses[label] = build_loss(loss).to(
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )  # pyright: ignore[reportCallIssue] # https://github.com/pytorch/pytorch/issues/131765

        assert self.losses, "At least one loss must be defined."

        # setup batch augmentations
        self.setup_batchaug()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self) -> None:
        train_opt = self.opt.train
        assert train_opt is not None
        assert train_opt.optim_ae is not None
        optim_params = []
        for k, v in self.net_ae.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning("Params %s will not be optimized.", k)

        optim_type = train_opt.optim_ae.pop("type")
        self.optimizer_ae = self.get_optimizer(
            optim_type, optim_params, **train_opt.optim_ae
        )
        self.optimizers.append(self.optimizer_ae)
        self.optimizers_skipped.append(False)
        self.optimizers_schedule_free.append("ScheduleFree" in optim_type)

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

    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        assert self.optimizer_ae is not None
        assert self.gt is not None
        assert self.scaler_ae is not None

        n_samples = self.gt.shape[0]
        self.loss_samples += n_samples

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            self.output_lq = self.net_ae.encode(self.gt)
            self.output_gt = self.net_ae.decode(self.output_lq)
            assert isinstance(self.output_gt, Tensor)
            l_ae_total = torch.tensor(0.0, device=self.output_gt.device)
            loss_dict = OrderedDict()

            for label, loss in self.losses.items():
                for output_type in ["gt", "lq"]:
                    l_ae_loss = loss(
                        getattr(self, f"output_{output_type}"),
                        getattr(self, output_type),
                    )
                    l_ae_total += l_ae_loss / self.accum_iters
                    loss_dict[f"{label}_{output_type}"] = l_ae_loss

            # add total generator loss for tensorboard tracking
            loss_dict["l_ae_total"] = l_ae_total

        self.scaler_ae.scale(l_ae_total).backward()
        if apply_gradient:
            if self.grad_clip:
                self.scaler_ae.unscale_(self.optimizer_ae)
                clip_grad_norm_(self.net_ae.parameters(), 1.0)

            scale_before = self.scaler_ae.get_scale()
            self.scaler_ae.step(self.optimizer_ae)
            self.scaler_ae.update()
            self.optimizers_skipped[0] = self.scaler_ae.get_scale() < scale_before
            self.optimizer_ae.zero_grad()

        for key, value in loss_dict.items():
            val = (
                value
                if isinstance(value, float)
                else value.to(dtype=torch.float32).detach()
            )
            self.log_dict[key] = self.log_dict.get(key, 0) + val * n_samples

        if self.net_ae_ema is not None and apply_gradient:
            if not (self.use_amp and self.optimizers_skipped[0]):
                self.net_ae_ema.update_parameters(self.net_ae)

    def test(self) -> None:
        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            if self.net_ae_ema is not None:
                self.net_ae_ema.eval()
                with torch.inference_mode():
                    self.output_lq = self.net_ae_ema.module.encode(self.gt)
                    self.output_gt = self.net_ae_ema.module.decode(self.output_lq)
            else:
                self.net_ae.eval()
                with torch.inference_mode():
                    self.output_lq = self.net_ae.encode(self.gt)
                    self.output_gt = self.net_ae.decode(self.output_lq)
                self.net_ae.train()

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

        for val_data in dataloader:
            img_name = osp.splitext(osp.basename(val_data["gt_path"][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            ae_img_lq = tensor2img(visuals["result_lq"])
            metric_data["img"] = ae_img_lq
            if "gt" in visuals:
                gt_img = tensor2img(visuals["gt"])
                metric_data[gt_key] = gt_img
                self.gt = None

            # tentative for out of GPU memory
            self.gt = None
            self.output_lq = None
            self.output_gt = None
            torch.cuda.empty_cache()

            save_img_dir = None

            if save_img:
                if self.opt.is_train:
                    if multi_val_datasets:
                        save_img_dir = osp.join(
                            self.opt.path.visualization, f"{dataset_name} - {img_name}"
                        )
                    else:
                        assert dataloader.dataset.opt.dataroot_gt is not None, (
                            "dataroot_gt is required for val set"
                        )
                        gt_path = val_data["gt_path"][0]

                        # multiple root paths are supported, find the correct root path for each gt_path
                        normalized_gt_path = osp.normpath(gt_path)

                        matching_root = None
                        for root in dataloader.dataset.opt.dataroot_gt:
                            normalized_root = osp.normpath(root)
                            if normalized_gt_path.startswith(normalized_root + osp.sep):
                                matching_root = root
                                break

                        if matching_root is None:
                            raise ValueError(
                                f"The gt_path {gt_path} does not match any of the provided dataroot_gt paths."
                            )

                        save_img_dir = osp.join(
                            self.opt.path.visualization,
                            osp.relpath(
                                osp.splitext(gt_path)[0],
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
                imwrite(ae_img_lq, save_img_path)  # TODO lq and gt
                if (
                    self.opt.is_train
                    and not self.first_val_completed
                    and "gt_path" in val_data
                ):
                    assert save_img_dir is not None
                    gt_img_target_path = osp.join(save_img_dir, f"{img_name}_gt.png")
                    if not os.path.exists(gt_img_target_path):
                        shutil.copy(val_data["gt_path"][0], gt_img_target_path)

            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")
        if pbar is not None:
            pbar.close()

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
        assert self.output_lq is not None
        assert self.output_gt is not None
        assert self.gt is not None

        out_dict = OrderedDict()
        out_dict["gt"] = self.gt.detach().cpu()
        out_dict["result_lq"] = self.output_lq.detach().cpu()
        out_dict["result_gt"] = self.output_gt.detach().cpu()

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

        if self.net_ae_ema is not None:
            self.save_network(
                self.net_ae_ema,
                "net_ae_ema",
                self.opt.path.models,
                current_iter,
                "params_ema",
            )

            self.save_network(
                self.net_ae,
                "net_ae",
                self.opt.path.resume_models,
                current_iter,
                "params",
            )
        else:
            self.save_network(
                self.net_ae, "net_ae", self.opt.path.models, current_iter, "params"
            )

        self.save_training_state(epoch, current_iter)
