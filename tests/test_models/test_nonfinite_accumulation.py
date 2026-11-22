"""With gradient accumulation, one non-finite microbatch skips its whole update, so
the non-finite counters must count updates: a run where every update has one bad
microbatch has to reach the abort, and the skipped updates' gradient norm and loss
scale must stay out of the logged averages."""

import sys
from os import path as osp
from pathlib import Path

import pytest
import torch
import yaml
from pytest import MonkeyPatch
from traiNNer.models.sr_model import SRModel
from traiNNer.utils.config import Config

ROOT = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))


def _model(tmp_path: Path, monkeypatch: MonkeyPatch) -> SRModel:
    with open(osp.join(ROOT, "options/_templates/train/SPAN/SPAN_fidelity.yml")) as f:
        cfg = yaml.safe_load(f)
    cfg["datasets"]["train"]["accum_iter"] = 2
    cfg["datasets"].pop("val", None)
    cfg["val"]["val_enabled"] = False
    cfg["use_amp"] = False
    cfg_path = tmp_path / "nonfinite.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["", "-opt", str(cfg_path)])
    opt, _ = Config.load_config_from_file(ROOT, is_train=True)
    opt.num_gpu = 0
    opt.path.pretrain_network_g = None
    return SRModel(opt)


def _window(model: SRModel, update: int) -> None:
    good = {"lq": torch.rand(2, 3, 16, 16), "gt": torch.rand(2, 3, 64, 64)}
    bad = {"lq": good["lq"].clone(), "gt": good["gt"].clone()}
    bad["lq"][0, 0, 0, 0] = float("nan")
    model.feed_data(bad)  # pyright: ignore[reportArgumentType]
    model.optimize_parameters(update, 1, False)
    model.feed_data(good)  # pyright: ignore[reportArgumentType]
    model.optimize_parameters(update, 0, True)


def test_skipped_updates_are_counted_and_abort(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # one test: the parsed config is a process-wide singleton, and building a second
    # model from it in the same process fails in setup_schedulers
    model = _model(tmp_path, monkeypatch)
    weights = [p.detach().clone() for p in model.net_g.parameters()]
    for update in range(1, 4):
        _window(model, update)
    assert int(model.nonfinite_streak) == 3
    assert int(model.nonfinite_steps) == 3
    for before, after in zip(weights, model.net_g.parameters(), strict=True):
        assert torch.equal(before, after)
    # the skipped updates' gradient norm stays out of the average
    log = model.get_current_log()
    assert torch.isfinite(torch.as_tensor(log["grad_norm_g"]))
    model.reset_current_log()
    # every update keeps being skipped, so the streak passes 10 and training stops
    for update in range(4, 13):
        _window(model, update)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        model.check_nonfinite_losses()
