import shutil
from pathlib import Path

import pytest
from traiNNer.archs.arch_info import ALL_ARCHS, arch_folder_name, template_filename
from traiNNer.utils.options import create_train_config, yaml_load

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A minimal repo root containing only the SPAN training templates."""
    src = REPO_ROOT / "options/_templates/train/SPAN"
    shutil.copytree(src, tmp_path / "options/_templates/train/SPAN")
    return tmp_path


def test_every_variant_has_all_training_templates() -> None:
    missing = [
        f"{arch_folder_name(arch)}/{template_filename(variant, otf, fromscratch)}"
        for arch in ALL_ARCHS
        for variant in arch["names"]
        for otf in (False, True)
        for fromscratch in (False, True)
        if not (
            REPO_ROOT
            / "options/_templates/train"
            / arch_folder_name(arch)
            / template_filename(variant, otf, fromscratch)
        ).is_file()
    ]
    assert missing == []


def test_variant_names_are_unique_case_insensitively() -> None:
    names = [n.lower() for arch in ALL_ARCHS for n in arch["names"]]
    assert len(names) == len(set(names))


def test_creates_config_with_name_and_scale(root: Path) -> None:
    out = create_train_config(root, "SPAN_s", scale=2, name="2x_test")

    assert out == root / "options/train/SPAN/2x_test.yml"
    template = (root / "options/_templates/train/SPAN/SPAN_S_fidelity.yml").read_text()
    created = out.read_text().splitlines()
    assert created[0] == (
        "# Created from options/_templates/train/SPAN/SPAN_S_fidelity.yml "
        "by train.py --new"
    )
    # Only the name and scale lines differ; the scale comment is kept.
    changed = [
        (a, b)
        for a, b in zip(template.splitlines(), created[1:], strict=True)
        if a != b
    ]
    assert changed == [
        ("name: 4x_SPAN_S", "name: 2x_test"),
        ("scale: 4  # 1, 2, 3, 4, 8", "scale: 2  # 1, 2, 3, 4, 8"),
    ]

    opt, _ = yaml_load(str(out))
    assert opt.name == "2x_test"
    assert opt.scale == 2


@pytest.mark.parametrize(
    ("gan", "otf", "template"),
    [
        (False, False, "SPAN_fidelity.yml"),
        (True, False, "SPAN_gan.yml"),
        (False, True, "SPAN_OTF_fidelity.yml"),
        (True, True, "SPAN_OTF_gan.yml"),
    ],
)
def test_gan_and_otf_select_template(
    root: Path, gan: bool, otf: bool, template: str
) -> None:
    out = create_train_config(root, "span", name="test", gan=gan, otf=otf)
    assert template in out.read_text().splitlines()[0]
    opt, _ = yaml_load(str(out))
    assert opt.high_order_degradation == otf


def test_default_name_follows_scale(root: Path) -> None:
    out = create_train_config(root, "span", scale=2, otf=True, gan=True)
    assert out.name == "2x_SPAN_OTF_finetune.yml"


def test_rejects_unsupported_scale(root: Path) -> None:
    with pytest.raises(ValueError, match="does not support scale 5"):
        create_train_config(root, "span", scale=5)


def test_unknown_variant_suggests_close_matches(root: Path) -> None:
    with pytest.raises(ValueError, match="Did you mean: span_s"):
        create_train_config(root, "spn_s")


def test_rejects_names_with_paths(root: Path) -> None:
    with pytest.raises(ValueError, match="Invalid name"):
        create_train_config(root, "span", name="../escape")


def test_does_not_overwrite(root: Path) -> None:
    create_train_config(root, "span", name="test")
    with pytest.raises(FileExistsError):
        create_train_config(root, "span", name="test")
