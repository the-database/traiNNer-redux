import os
import time
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import pytorch_optimizer
import torch
from safetensors.torch import load_file, save_file
from spandrel import ModelLoader
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import ParamsT
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from ..ops.batchaug import MOA_DEBUG_PATH, BatchAugment
from ..utils import get_root_logger
from ..utils.dist_util import master_only
from ..utils.types import DataFeed, TrainingState


class BaseModel:
    """Base model."""

    def __init__(self, opt: dict[str, Any]) -> None:
        self.opt = opt
        self.device = torch.device("cuda" if opt["num_gpu"] != 0 else "cpu")
        self.is_train = opt["is_train"]
        self.schedulers: list[LRScheduler] = []
        self.optimizers: list[Optimizer] = []
        self.batch_augment = None
        self.log_dict = {}
        self.loss_samples = 0
        self.metric_results: dict[str, Any] = {}
        self.best_metric_results: dict[str, Any] = {}
        self.model_loader = ModelLoader()
        self.net_g = None
        self.net_g_ema = None
        self.net_d = None
        self.use_amp = False
        self.amp_dtype = torch.float16
        self.scaler_g: GradScaler | None = None
        self.scaler_d: GradScaler | None = None

    @abstractmethod
    def feed_data(self, data: DataFeed) -> None:
        pass

    @abstractmethod
    def optimize_parameters(self, current_iter: int) -> None:
        pass

    @abstractmethod
    def get_current_visuals(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def save(self, epoch: int, current_iter: int) -> None:
        """Save networks and training state."""

    def validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool = False,
    ) -> None:
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt["dist"]:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    @abstractmethod
    def dist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
    ) -> None:
        pass

    @abstractmethod
    def nondist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
    ) -> None:
        pass

    def _initialize_best_metric_results(self, dataset_name: str) -> None:
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if dataset_name in self.best_metric_results:
            return

        # add a dataset record
        record = {}
        for metric, content in self.opt["val"]["metrics"].items():
            better = content.get("better", "higher")
            init_val = float("-inf") if better == "higher" else float("inf")
            record[metric] = {"better": better, "val": init_val, "iter": -1}
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(
        self, dataset_name: str, metric: str, val: float, current_iter: int
    ) -> None:
        if self.best_metric_results[dataset_name][metric]["better"] == "higher":
            if val >= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter
        elif val <= self.best_metric_results[dataset_name][metric]["val"]:
            self.best_metric_results[dataset_name][metric]["val"] = val
            self.best_metric_results[dataset_name][metric]["iter"] = current_iter

    def model_ema(self, decay: float = 0.999) -> None:
        assert self.net_g is not None
        assert self.net_g_ema is not None

        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(
                net_g_params[k].data, alpha=1 - decay
            )

    def get_current_log(self) -> dict[str, float | torch.Tensor]:
        return {k: v / self.loss_samples for k, v in self.log_dict.items()}

    def reset_current_log(self) -> None:
        self.log_dict = {}
        self.loss_samples = 0

    def model_to_device(self, net: nn.Module) -> nn.Module:
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt["dist"]:
            find_unused_parameters = self.opt.get("find_unused_parameters", False)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters,
            )
        elif self.opt["num_gpu"] > 1:
            net = DataParallel(net)
        return net

    def get_optimizer(
        self,
        optim_type: str,
        params: ParamsT,
        lr: float,
        **kwargs,
    ) -> Optimizer:
        if optim_type == "AdamP":
            optimizer = pytorch_optimizer.AdamP(params, lr, **kwargs)
        elif optim_type == "Lamb":
            optimizer = pytorch_optimizer.Lamb(params, lr, **kwargs)
        elif optim_type == "Prodigy":
            optimizer = pytorch_optimizer.Prodigy(params, lr, **kwargs)
        elif optim_type == "Lion":
            optimizer = pytorch_optimizer.Lion(params, lr, **kwargs)
        elif optim_type == "Tiger":
            optimizer = pytorch_optimizer.Tiger(params, lr, **kwargs)
        elif optim_type == "Adan":
            optimizer = pytorch_optimizer.Adan(params, lr, **kwargs)
        elif optim_type == "Adam":
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == "AdamW":
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == "Adamax":
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == "SGD":
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == "ASGD":
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == "RMSprop":
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == "Rprop":
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supported yet.")
        return optimizer

    def setup_schedulers(self) -> None:
        # https://github.com/Corpsecreate/neosr/blob/a29e509dae5cd39aea94ac82d1347d2a54e1175c/neosr/models/default.py#L276

        """Set up schedulers."""
        train_opt = self.opt["train"]
        scheduler_type = train_opt["scheduler"].pop("type")
        # uppercase scheduler_type to make it case insensitive
        sch_typ_upper = scheduler_type.upper()
        sch_map: dict[str, type[LRScheduler]] = {
            "CONSTANTLR": torch.optim.lr_scheduler.ConstantLR,
            "LINEARLR": torch.optim.lr_scheduler.LinearLR,
            "EXPONENTIALLR": torch.optim.lr_scheduler.ExponentialLR,
            "CYCLICLR": torch.optim.lr_scheduler.CyclicLR,
            "STEPLR": torch.optim.lr_scheduler.StepLR,
            "MULTISTEPLR": torch.optim.lr_scheduler.MultiStepLR,
            "LAMBDALR": torch.optim.lr_scheduler.LambdaLR,
            "MULTIPLICATIVELR": torch.optim.lr_scheduler.MultiplicativeLR,
            "SEQUENTIALLR": torch.optim.lr_scheduler.SequentialLR,
            "CHAINEDSCHEDULER": torch.optim.lr_scheduler.ChainedScheduler,
            "ONECYCLELR": torch.optim.lr_scheduler.OneCycleLR,
            "POLYNOMIALLR": torch.optim.lr_scheduler.PolynomialLR,
            "COSINEANNEALINGWARMRESTARTS": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            "COSINEANNEALING": torch.optim.lr_scheduler.CosineAnnealingLR,
            "REDUCELRONPLATEAU": torch.optim.lr_scheduler.ReduceLROnPlateau,
        }
        if sch_typ_upper in sch_map:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    sch_map[sch_typ_upper](optimizer, **train_opt["scheduler"])
                )
        else:
            raise NotImplementedError(
                f"Scheduler {scheduler_type} is not implemented yet."
            )

    def get_bare_model(
        self, net: DataParallel | DistributedDataParallel | nn.Module
    ) -> nn.Module:
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, DataParallel | DistributedDataParallel):
            net = net.module
        return net

    @master_only
    def print_network(self, net: nn.Module) -> None:
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, DataParallel | DistributedDataParallel):
            net_cls_str = f"{net.__class__.__name__} - {net.module.__class__.__name__}"
        else:
            net_cls_str = f"{net.__class__.__name__}"

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(x.numel() for x in net.parameters())

        logger = get_root_logger()
        logger.info("Network: %s, with parameters: %s", net_cls_str, f"{net_params:,d}")
        logger.info(net_str)

    def _set_lr(self, lr_groups_l: list[list[float]]) -> None:
        """Set learning rate for warm-up.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l, strict=False):
            for param_group, lr in zip(optimizer.param_groups, lr_groups, strict=False):
                param_group["lr"] = lr

    def _get_init_lr(self) -> list[list[float]]:
        """Get the initial lr, which is set by the scheduler."""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter: int, warmup_iter: int = -1) -> None:
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int): Warm-up iter numbers. -1 for no warm-up.
                Default: -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self) -> list[float]:
        return [param_group["lr"] for param_group in self.optimizers[0].param_groups]

    @master_only
    def save_network(
        self,
        net: nn.Module,
        net_label: str,
        current_iter: int,
        param_key: str = "params",
    ) -> None:
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """

        current_iter_str = "latest" if current_iter == -1 else str(current_iter)

        save_filename = f"{net_label}_{current_iter_str}.safetensors"
        save_path = os.path.join(self.opt["path"]["models"], save_filename)

        bare_net_ = self.get_bare_model(net)
        state_dict = bare_net_.state_dict()
        for full_key, param in state_dict.items():
            key = full_key
            if key.startswith("module."):  # remove unnecessary 'module.'
                key = key[7:]
            state_dict[key] = param.cpu()

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                save_file(state_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(
                    "Save model error: %s, remaining retry times: %d", e, retry - 1
                )
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning("Still cannot save %s. Just ignore it.", save_path)
            # raise IOError(f'Cannot save {save_path}.')

    def _print_different_keys_loading(
        self,
        crt_net: nn.Module,
        load_net: dict[str, Any],
        file_path: str,
        strict: bool = True,
    ) -> None:
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net_state_dict = crt_net.state_dict()
        crt_net_keys = set(crt_net_state_dict.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning("Current net - loaded net:")
            for v in sorted(crt_net_keys - load_net_keys):
                logger.warning("  %s", v)
            logger.warning("Loaded net - current net:")
            for v in sorted(load_net_keys - crt_net_keys):
                logger.warning("  %s", v)

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net_state_dict[k].size() != load_net[k].size():
                    logger.warning(
                        "Size different, ignore [%s]: crt_net: " "%s; load_net: %s",
                        k,
                        crt_net_state_dict[k].shape,
                        load_net[k].shape,
                    )
                    load_net[k + ".ignore"] = load_net.pop(k)

            new_common_keys = crt_net_keys & set(load_net.keys())
            if len(new_common_keys) == 0:
                logger.warning(
                    "Pretrain model %s matched %.2f%% of keys of currently training model. Pretrain will have no effect and the model will be trained from scratch.",
                    file_path,
                    0,
                )
            elif len(new_common_keys) == len(crt_net_keys):
                logger.info(
                    "Pretrain model %s matched %.2f%% of keys of currently training model. Pretrain is loaded in strict mode.",
                    file_path,
                    1,
                )
            else:
                overlap = len(new_common_keys) / len(crt_net_keys) * 100
                logger.info(
                    "Pretrain model %s matched %.2f%% of keys of currently training model.",
                    file_path,
                    overlap,
                )

    def load_network_spandrel(
        self, net: nn.Module, load_path: str, strict: bool = True
    ) -> bool | None:
        try:
            logger = get_root_logger()
            load_net = self.model_loader.load_from_file(load_path)
            net.load_state_dict(load_net.model.state_dict(), strict=strict)
            logger.info(
                "Loading %s model from %s, with spandrel.",
                net.__class__.__name__,
                load_path,
            )
            return True
        except Exception as e:
            print(e)
            return False

    def load_network(
        self,
        net: nn.Module,
        load_path: str,
        strict: bool = True,
        param_key: str | None = "params",
    ) -> None:
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """

        logger = get_root_logger()
        try:
            load_net = self.model_loader.load_from_file(load_path).model.state_dict()
            # net.load_state_dict(load_net.model.state_dict(), strict=strict)
            logger.info(
                "Loading %s model from %s, with spandrel.",
                net.__class__.__name__,
                load_path,
            )
        except Exception as e:
            print(e)

            net = self.get_bare_model(net)
            if load_path.endswith(".safetensors"):
                load_net = load_file(load_path, device=str(self.device))
            elif load_path.endswith(".pth"):
                load_net = torch.load(
                    load_path, map_location=lambda storage, loc: storage
                )

                if param_key is not None:
                    if param_key not in load_net:
                        if "params_ema" in load_net:
                            logger.info(
                                "Loading: %s does not exist, using params_ema.",
                                param_key,
                            )
                            param_key = "params_ema"
                        elif "params" in load_net:
                            logger.info(
                                "Loading: %s does not exist, using params.", param_key
                            )
                            param_key = "params"
                        else:
                            logger.info(
                                "Loading: %s does not exist, using None.", param_key
                            )
                            param_key = None
                    else:
                        load_net = load_net[param_key]
            else:
                raise ValueError(f"Unsupported model: {load_path}") from e
            logger.info(
                "Loading %s model from %s, with param key: [%s].",
                net.__class__.__name__,
                load_path,
                param_key,
            )
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith("module."):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, load_path, strict)

        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch: int, current_iter: int) -> None:
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            assert self.scaler_g is not None
            assert self.scaler_d is not None
            state: TrainingState = {
                "epoch": epoch,
                "iter": current_iter,
                "optimizers": [],
                "schedulers": [],
            }

            if self.use_amp:
                state["scaler_d"] = self.scaler_d.state_dict()
                state["scaler_g"] = self.scaler_g.state_dict()

            for o in self.optimizers:
                state["optimizers"].append(o.state_dict())
            for s in self.schedulers:
                state["schedulers"].append(s.state_dict())
            save_filename = f"{current_iter}.state"
            save_path = os.path.join(self.opt["path"]["training_states"], save_filename)

            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(
                        "Save training state error: %s, remaining retry times: %d",
                        e,
                        retry - 1,
                    )
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning("Still cannot save %s. Just ignore it.", save_path)
                # raise IOError(f'Cannot save {save_path}.')

    def resume_training(self, resume_state: TrainingState) -> None:
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        assert self.scaler_d is not None
        assert self.scaler_g is not None

        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]

        assert len(resume_optimizers) == len(
            self.optimizers
        ), "Wrong lengths of optimizers"
        assert len(resume_schedulers) == len(
            self.schedulers
        ), "Wrong lengths of schedulers"

        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        if "scaler_g" in resume_state:
            self.scaler_g.load_state_dict(resume_state["scaler_g"])
        if "scaler_d" in resume_state:
            self.scaler_d.load_state_dict(resume_state["scaler_d"])

    def reduce_loss_dict(self, loss_dict: dict[str, Any]) -> OrderedDict[str, Any]:
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt["dist"]:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)  # type: ignore
                if self.opt["rank"] == 0:
                    losses /= self.opt["world_size"]
                loss_dict = dict(zip(keys, losses, strict=False))

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict

    def setup_batchaug(self) -> None:
        train_opt = self.opt["train"]
        logger = get_root_logger()
        if train_opt.get("use_moa", False):
            self.batch_augment = BatchAugment(train_opt)
            logger.info(
                "Mixture of augmentations (MoA) enabled, with augs: %s and probs: %s",
                self.batch_augment.moa_augs,
                self.batch_augment.moa_probs,
            )
            if self.batch_augment.debug:
                logger.info(
                    "MoA debugging enabled. Augmented tiles will be saved to: %s",
                    MOA_DEBUG_PATH,
                )
