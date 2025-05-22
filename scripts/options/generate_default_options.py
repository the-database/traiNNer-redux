import os
import re
from os import path as osp
from typing import Any

from traiNNer.archs.arch_info import (
    ALL_ARCHS,
    ARCHS_WITHOUT_CHANNELS_LAST,
    ARCHS_WITHOUT_FP16,
    OFFICIAL_SETTINGS_FINETUNE,
    OFFICIAL_SETTINGS_FROMSCRATCH,
    ArchInfo,
)


def template_filename(variant: str, otf: bool, fromscratch: bool) -> str:
    return f"{variant}{'_OTF' if otf else ''}_{'fromscratch' if fromscratch else 'finetune'}.yml"


def final_template(
    template: str,
    arch: ArchInfo,
    variant: str,
    training_settings: dict[str, dict[str, Any]] | None = None,
    template_otf1: str = "",
    template_otf2: str = "",
    name_suffix: str = "",
    mssim_only: bool = False,
) -> str:
    default_scale = 4

    if arch["scales"] == [1]:
        default_scale = 1

        # PSNR loss instead of charbonnier loss
        template = template.replace(
            """# Charbonnier loss
    - type: charbonnierloss""",
            """# PSNR loss
    - type: psnrloss""",
        )

        # switch MultiStepLR to CosineAnnealingLR
        template = template.replace(
            """scheduler:
    type: MultiStepLR
    milestones: %milestones%
    gamma: 0.5""",
            """scheduler:
    type: CosineAnnealingLR
    T_max: %t_max%
    eta_min: %eta_min%""",
        )

    template = template.replace(
        "scale: %scale%",
        f"scale: {default_scale}  # {', '.join([str(x) for x in arch['scales']])}",
    )

    arch_type_str = f"type: {variant}"
    # if len(arch["names"]) > 1:
    #     arch_type_str += f"  # {', '.join([str(x) for x in arch['names']])}"

    if "extras" in arch:
        for k, v in arch["extras"].items():
            arch_type_str += f"\n  {k}: {v}"

    if name_suffix:
        template = template.replace(
            "name: %scale%_%archname%", f"name: %scale%_%archname%_{name_suffix}"
        )

    template = template.replace(
        "type: %archname%",
        arch_type_str.lower(),
    )

    if arch["names"][0].lower() in ARCHS_WITHOUT_FP16:
        template = template.replace("amp_bf16: false", "amp_bf16: true")

    arch_key = variant.lower()

    # print(
    #     arch_key,
    #     arch_key in ARCHS_WITHOUT_CHANNELS_LAST,
    # )

    if arch_key in ARCHS_WITHOUT_CHANNELS_LAST:
        template = template.replace(
            "use_channels_last: true", "use_channels_last: false"
        )

    if training_settings is not None:
        settings = training_settings.get(arch_key, training_settings[""])
        if arch_key not in training_settings:
            # print(arch_key)
            pass
        for name, value in settings.items():
            # print("training settings", arch_key, name, value)
            template = template.replace(f"%{name}%", str(value))

    # defaults
    template = template.replace("%betas%", "[0.9, 0.99]")  # adamw betas
    template = template.replace("%ema_decay%", "0.999")  # adamw betas

    template = template.replace("%archname%", f"{variant}")
    template = template.replace("%scale%", f"{default_scale}x")

    if "overrides" in arch:
        for k, v in arch["overrides"].items():
            template = re.sub(rf"{k}: .+", rf"{k}: {v}", template)

    if "pth_override" in arch:
        template = template.replace(
            "save_checkpoint_format: safetensors", "save_checkpoint_format: pth"
        )

    template = template.replace("%otf1%\n", template_otf1)
    template = template.replace("    %otf2%\n", template_otf2)

    if template_otf1 and template_otf2:
        template = template.replace("%traindatasettype%", "realesrgandataset")
        template = template.replace(
            "    # Path to the LR (low res) images in your training dataset.\n", ""
        )
        template = template.replace(
            "    dataroot_lq: [\n      datasets/train/dataset1/lr,\n      datasets/train/dataset1/lr2\n    ]\n",
            "",
        )
    else:
        template = template.replace("%traindatasettype%", "pairedimagedataset")

    if "video_override" in arch:
        template = template.replace(
            "type: pairedimagedataset", "type: pairedvideodataset"
        )

        template = template.replace(
            "type: singleimagedataset", "type: singlevideodataset"
        )

        template = re.sub(
            r"(\s+)(dataroot_lq:\s*\[\s*[^]]*])", r"\1\2\1clip_size: 5", template
        )

    if mssim_only:
        template = re.sub("loss_weight: [0-9.]+", "loss_weight: 0", template)
        template = template.replace(
            "type: msssiml1loss\n      alpha: 0.1\n      loss_weight: 0",
            "type: msssiml1loss\n      alpha: 0.1\n      loss_weight: 1.0",
        )

    return template


template_path_paired_fromscratch = osp.normpath(
    osp.join(
        __file__,
        osp.pardir,
        "./train_default_options_paired_fromscratch.yml",
    )
)

template_path_paired_finetune = osp.normpath(
    osp.join(__file__, osp.pardir, "./train_default_options_paired_finetune.yml")
)

template_path_otf1 = osp.normpath(
    osp.join(__file__, osp.pardir, "./train_default_options_otf1.yml")
)

template_path_otf2 = osp.normpath(
    osp.join(__file__, osp.pardir, "./train_default_options_otf2.yml")
)

template_path_single = osp.normpath(
    osp.join(__file__, osp.pardir, "./test_default_options_single.yml")
)

template_path_onnx = osp.normpath(
    osp.join(__file__, osp.pardir, "./onnx_default_options.yml")
)

with (
    open(template_path_paired_fromscratch) as fps,
    open(template_path_paired_finetune) as fpf,
    open(template_path_otf1) as fo1,
    open(template_path_otf2) as fo2,
    open(template_path_single) as fts,
    open(template_path_onnx) as fox,
):
    template_paired_fromscratch = fps.read()
    template_paired_finetune = fpf.read()
    template_otf1 = fo1.read()
    template_otf2 = fo2.read()
    template_test_single = fts.read()
    template_onnx = fox.read()


if __name__ == "__main__":
    for arch in ALL_ARCHS:
        for variant in arch["names"]:
            folder_name = arch["names"][0].split("_")[0]

            if "folder_name_override" in arch:
                folder_name = arch["folder_name_override"]

            train_folder_path = osp.normpath(
                osp.join(
                    __file__,
                    osp.pardir,
                    osp.pardir,
                    osp.pardir,
                    "./options/_templates/train",
                    folder_name,
                )
            )
            test_folder_path = osp.normpath(
                osp.join(
                    __file__,
                    osp.pardir,
                    osp.pardir,
                    osp.pardir,
                    "./options/_templates/test",
                    folder_name,
                )
            )
            onnx_folder_path = osp.normpath(
                osp.join(
                    __file__,
                    osp.pardir,
                    osp.pardir,
                    osp.pardir,
                    "./options/_templates/onnx",
                    folder_name,
                )
            )

            train_folder_path2 = osp.normpath(
                osp.join(
                    __file__,
                    osp.pardir,
                    osp.pardir,
                    osp.pardir,
                    "./options/train",
                    folder_name,
                )
            )
            test_folder_path2 = osp.normpath(
                osp.join(
                    __file__,
                    osp.pardir,
                    osp.pardir,
                    osp.pardir,
                    "./options/test",
                    folder_name,
                )
            )
            onnx_folder_path2 = osp.normpath(
                osp.join(
                    __file__,
                    osp.pardir,
                    osp.pardir,
                    osp.pardir,
                    "./options/onnx",
                    folder_name,
                )
            )

            os.makedirs(train_folder_path, exist_ok=True)
            os.makedirs(test_folder_path, exist_ok=True)
            os.makedirs(onnx_folder_path, exist_ok=True)
            os.makedirs(train_folder_path2, exist_ok=True)
            os.makedirs(test_folder_path2, exist_ok=True)
            os.makedirs(onnx_folder_path2, exist_ok=True)

            with open(
                osp.join(
                    train_folder_path,
                    template_filename(variant, otf=False, fromscratch=True),
                ),
                mode="w",
            ) as fw:
                fw.write(
                    final_template(
                        template_paired_fromscratch,
                        arch,
                        variant,
                        OFFICIAL_SETTINGS_FROMSCRATCH,
                    )
                )

            with open(
                osp.join(
                    train_folder_path,
                    template_filename(variant, otf=False, fromscratch=False),
                ),
                mode="w",
            ) as fw:
                fw.write(
                    final_template(
                        template_paired_finetune,
                        arch,
                        variant,
                        OFFICIAL_SETTINGS_FINETUNE,
                    )
                )

            with open(
                osp.join(
                    train_folder_path,
                    template_filename(variant, otf=True, fromscratch=True),
                ),
                mode="w",
            ) as fw:
                fw.write(
                    final_template(
                        template_paired_fromscratch,
                        arch,
                        variant,
                        OFFICIAL_SETTINGS_FROMSCRATCH,
                        template_otf1,
                        template_otf2,
                        "OTF_fromscratch",
                    )
                )

            with open(
                osp.join(
                    train_folder_path,
                    template_filename(variant, otf=True, fromscratch=False),
                ),
                mode="w",
            ) as fw:
                fw.write(
                    final_template(
                        template_paired_finetune,
                        arch,
                        variant,
                        OFFICIAL_SETTINGS_FINETUNE,
                        template_otf1,
                        template_otf2,
                        "OTF_finetune",
                    )
                )

            with open(osp.join(test_folder_path, f"{variant}.yml"), mode="w") as fw:
                fw.write(final_template(template_test_single, arch, variant))

            with open(osp.join(onnx_folder_path, f"{variant}.yml"), mode="w") as fw:
                fw.write(
                    final_template(
                        template_onnx,
                        arch,
                        variant,
                        None,
                        template_otf1,
                        template_otf2,
                    )
                )

            gitignore_contents = """# Ignore everything in this directory
*
# Except this file
!.gitignore
"""

            with open(
                osp.join(
                    train_folder_path2,
                    ".gitignore",
                ),
                mode="w",
            ) as fw:
                fw.write(gitignore_contents)

            with open(
                osp.join(
                    test_folder_path2,
                    ".gitignore",
                ),
                mode="w",
            ) as fw:
                fw.write(gitignore_contents)

            with open(
                osp.join(
                    onnx_folder_path2,
                    ".gitignore",
                ),
                mode="w",
            ) as fw:
                fw.write(gitignore_contents)
