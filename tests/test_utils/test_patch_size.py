import pytest
from traiNNer.utils.options import derive_train_patch_size, format_patch_size
from traiNNer.utils.redux_options import DatasetOptions


def make_opt(
    dataset_type: str = "pairedimagedataset",
    gt_size: int | tuple[int, int] | None = None,
    lq_size: int | tuple[int, int] | None = None,
) -> DatasetOptions:
    return DatasetOptions(
        name="train", type=dataset_type, gt_size=gt_size, lq_size=lq_size
    )


def test_square_lq_size_stays_int() -> None:
    opt = make_opt(lq_size=64)
    derive_train_patch_size(opt, 4, high_order_degradation=False)
    assert opt.gt_size == 256
    assert format_patch_size(opt.gt_size) == "256"


def test_square_gt_size_stays_int() -> None:
    opt = make_opt(gt_size=256)
    derive_train_patch_size(opt, 4, high_order_degradation=False)
    assert opt.lq_size == 64


@pytest.mark.parametrize("dataset_type", ["pairedimagedataset", "PairedVideoDataset"])
def test_rectangular_lq_size_scales_width_and_height(dataset_type: str) -> None:
    opt = make_opt(dataset_type, lq_size=(48, 32))
    derive_train_patch_size(opt, 2, high_order_degradation=False)
    assert opt.gt_size == (96, 64)
    assert format_patch_size(opt.gt_size) == "96x64"


def test_rectangular_gt_size_derives_lq_size() -> None:
    opt = make_opt(gt_size=(96, 64))
    derive_train_patch_size(opt, 2, high_order_degradation=False)
    assert opt.lq_size == (48, 32)


@pytest.mark.parametrize(
    ("gt_size", "lq_size"), [(None, None), (128, 32), ((128, 64), (32, 16))]
)
def test_requires_exactly_one_size(
    gt_size: int | tuple[int, int] | None, lq_size: int | tuple[int, int] | None
) -> None:
    opt = make_opt(gt_size=gt_size, lq_size=lq_size)
    with pytest.raises(ValueError, match="Exactly one of gt_size or lq_size"):
        derive_train_patch_size(opt, 4, high_order_degradation=False)


@pytest.mark.parametrize(
    ("dataset_type", "high_order_degradation"),
    [
        ("realesrgandataset", True),
        ("pairedimagedataset", True),
        ("singlegtdataset", False),
    ],
)
def test_rectangular_size_rejected_for_square_only_training(
    dataset_type: str, high_order_degradation: bool
) -> None:
    opt = make_opt(dataset_type, lq_size=(48, 32))
    with pytest.raises(ValueError, match=r"\[width, height\] rectangle"):
        derive_train_patch_size(opt, 2, high_order_degradation)
