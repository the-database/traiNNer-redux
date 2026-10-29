# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

traiNNer-redux trains PyTorch models for image/video super-resolution and restoration (a BasicSR descendant). Trained models are run in chaiNNer, so Spandrel weight compatibility matters.

## Commands

Activate the venv first: `source venv/Scripts/activate` (Git Bash on Windows).

```bash
pip install -e .[dev]          # add ,onnx for ONNX export deps

ruff format && ruff check --fix
pyright
pytest tests/
pytest tests/test_archs/test_archs.py -k "span"

python train.py --new span_s --scale 2 --name 2x_test   # creates options/train/SPAN/2x_test.yml from a template
python train.py -opt options/train/SPAN/2x_test.yml
python test.py -opt options/test/SPAN/SPAN_S.yml
python convert_to_onnx.py -opt options/onnx/SPAN/SPAN_S.yml
```

The checked-in sample dataset in `datasets/` lets shipped configs run without any data prep. Output is written to `experiments/<name>/` (train) or `results/<name>/` (test).

## Architecture

**Discovery by filename.** Every `*_arch.py`, `*_loss.py` and `*_dataset.py` is auto-imported and registers itself through a decorator from `traiNNer/utils/registry.py`. Registry keys are lowercased names, so `type: span_f64` resolves to the function `span_f64`.

**Arch registries.** `build_network` checks `SPANDREL_REGISTRY` first (state dicts must stay Spandrel-loadable), then `ARCH_REGISTRY` (traiNNer-only), then `TESTARCH_REGISTRY` (experimental, undocumented). Arch files are self-contained: never import from another `*_arch.py`; copy the shared code instead.

**Config.** YAML is decoded into strict msgspec Structs (`traiNNer/utils/redux_options.py`, `forbid_unknown_fields`), so every new YAML key needs a Struct field. Derived values are filled in by `traiNNer/utils/options.py`. The parsed config is a global singleton (`traiNNer/utils/config.py`). Loading a config from `options/_templates/` is rejected. User configs live in `options/train/<ARCH>/`, which is gitignored; `train.py --new` creates them from a template (`create_train_config` in `options.py`).

**Model class.** `build_model` chooses it from config flags, not from a `type` key: `high_order_degradation` → `RealESRGANModel`/`RealESRGANPairedModel`, `network_ae` alone → `AEModel`, otherwise `SRModel`.

**`traiNNer/archs/arch_info.py`** catalogs every arch: variants, scales, official training settings, and capability sets (`ARCHS_WITHOUT_FP16`, `ARCHS_WITHOUT_CHANNELS_LAST`, `REQUIRE_32_HW`/`REQUIRE_64_HW`, `TEMPORAL_ARCHS`). The options generator, tests and ONNX export all read it, so add an entry for each new arch. New archs are picked up by `tests/test_archs/test_archs.py` automatically; list exclusions and unsupported scales in that file.

## Generated files

Don't hand-edit these. Change the inputs and regenerate:

| Output | Generator | Inputs |
|---|---|---|
| `options/_templates/{train,test,onnx}/<ARCH>/*.yml` | `scripts/options/generate_default_options.py` | `scripts/options/*_default_options*.yml`, `arch_info.py` |
| `schemas/redux-config.schema.json` | `scripts/options/generate_schema.py` | `redux_options.py` |
| Reference docs in `docs/source/` | `scripts/options/generate_docs_from_{archs,losses,msgspec}.py` | registries, `redux_options.py` |
