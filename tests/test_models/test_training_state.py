import os
import stat
from pathlib import Path

import pytest
import torch
from traiNNer.models import base_model
from traiNNer.models.sr_model import SRModel
from traiNNer.utils.redux_options import PathOptions, ReduxOptions


def test_failed_write_preserves_the_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "3.state"
    path.write_bytes(b"complete")
    monkeypatch.setattr(base_model.time, "sleep", lambda _: None)

    def fail(temporary: str) -> None:
        Path(temporary).write_bytes(b"partial")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        base_model._write_checkpoint(path, fail)  # noqa: SLF001
    base_model._write_checkpoint(path, fail, optional=True)  # noqa: SLF001
    assert path.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name != "posix", reason="POSIX checkpoint permissions")
def test_checkpoint_respects_umask_and_existing_permissions(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.touch(mode=0o666)
    path = tmp_path / "3.state"

    def write(name: str) -> None:
        Path(name).write_bytes(b"weights")

    base_model._write_checkpoint(path, write)  # noqa: SLF001
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(ordinary.stat().st_mode)
    path.chmod(0o600)
    base_model._write_checkpoint(path, write)  # noqa: SLF001
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_log_averages_count_valid_samples_for_each_metric() -> None:
    model = SRModel(
        ReduxOptions(
            name="metrics",
            scale=2,
            num_gpu=0,
            path=PathOptions(),
            network_g={"type": "compact", "num_feat": 8, "num_conv": 1},
        )
    )
    model.accumulate_log({"loss": 2.0}, 4, torch.tensor(True))
    model.accumulate_log({"loss": float("nan")}, 4, torch.tensor(False))
    model.accumulate_log({"loss": 8.0, "scale": 1.0}, 2, torch.tensor(True))
    values = model.get_current_log()
    assert float(values["loss"]) == 4.0
    assert float(values["scale"]) == 1.0
    model.reset_current_log()
    assert model.get_current_log() == {}
