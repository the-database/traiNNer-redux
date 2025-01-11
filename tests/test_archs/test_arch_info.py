from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY
from traiNNer.archs.arch_info import (
    ARCHS_WITHOUT_CHANNELS_LAST,
    ARCHS_WITHOUT_FP16,
    OFFICIAL_SETTINGS_FINETUNE,
    OFFICIAL_SETTINGS_FROMSCRATCH,
)

ALL_ARCH_NAMES = {x[0] for x in list(ARCH_REGISTRY) + list(SPANDREL_REGISTRY)}


class TestArchInfo:
    def test_archs_without_fp16(
        self,
    ) -> None:
        failures = {name for name in ARCHS_WITHOUT_FP16 if name not in ALL_ARCH_NAMES}
        assert not failures, (
            f"The following ARCHS_WITHOUT_FP16 are missing from ALL_ARCH_NAMES: {', '.join(failures)}"
        )

    def test_archs_without_channels_last(
        self,
    ) -> None:
        failures = {
            name for name in ARCHS_WITHOUT_CHANNELS_LAST if name not in ALL_ARCH_NAMES
        }
        assert not failures, (
            f"The following ARCHS_WITHOUT_CHANNELS_LAST are missing from ALL_ARCH_NAMES: {', '.join(failures)}"
        )

    # def test_official_metrics(
    #     self,
    # ) -> None:
    #     failures = {name for name in OFFICIAL_METRICS if name not in ALL_ARCH_NAMES}
    #     assert not failures, f"The following OFFICIAL_METRICS are missing from ALL_ARCH_NAMES: {', '.join(failures)}"

    def test_official_settings_fromscratch(
        self,
    ) -> None:
        failures = {
            name
            for name in OFFICIAL_SETTINGS_FROMSCRATCH
            if name not in ALL_ARCH_NAMES
            if name
        }
        assert not failures, (
            f"The following OFFICIAL_SETTINGS_FROMSCRATCH are missing from ALL_ARCH_NAMES: {', '.join(failures)}"
        )

    def test_official_settings_finetune(
        self,
    ) -> None:
        failures = {
            name
            for name in OFFICIAL_SETTINGS_FINETUNE
            if name not in ALL_ARCH_NAMES
            if name
        }
        assert not failures, (
            f"The following OFFICIAL_SETTINGS_FINETUNE are missing from ALL_ARCH_NAMES: {', '.join(failures)}"
        )
