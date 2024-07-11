import logging
from os import path as osp

import torch
from torch.utils.data import DataLoader
from traiNNer.data import build_dataloader, build_dataset
from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.prefetch_dataloader import PrefetchDataLoader
from traiNNer.models import build_model
from traiNNer.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from traiNNer.utils.config import Config
from traiNNer.utils.options import dict2str, struct2dict


def test_pipeline(root_path: str) -> None:
    # parse options, set distributed setting, set ramdom seed
    opt, _ = Config.load_config_from_file(root_path, is_train=False)
    assert opt.val is not None
    assert opt.dist is not None
    assert opt.path.log is not None
    assert isinstance(opt.num_gpu, int)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt.path.log, f"test_{opt.name}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name="traiNNer", log_level=logging.INFO, log_file=log_file
    )
    logger.info(get_env_info())
    logger.info(dict2str(struct2dict(opt)))

    # create test dataset and dataloader
    test_loaders: list[PrefetchDataLoader | DataLoader] = []
    for _, dataset_opt in sorted(opt.datasets.items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt.num_gpu,
            dist=opt.dist,
            sampler=None,
            seed=opt.manual_seed,
        )
        logger.info("Number of test images in %s: %d", dataset_opt.name, len(test_set))
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        assert isinstance(test_loader.dataset, BaseDataset)
        test_set_name = test_loader.dataset.opt.name
        logger.info("Testing %s...", test_set_name)
        model.validation(
            test_loader,
            current_iter=-1,
            tb_logger=None,
            save_img=opt.val.save_img,
        )


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_pipeline(root_path)
