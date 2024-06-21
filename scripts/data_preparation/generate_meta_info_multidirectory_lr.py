import argparse
import os


def get_lr_hr_pair(hr_folder: str, lr_folder: str) -> list[tuple[str, str]]:
    hr_files = os.listdir(hr_folder)
    lr_files = os.listdir(lr_folder)

    hr_files.sort()
    lr_files.sort()

    lr_hr_pairs = []
    for hr_file in hr_files:
        base_name, _ = os.path.splitext(hr_file)
        if base_name + ".png" in lr_files:
            lr_hr_pairs.append(
                (
                    os.path.join(hr_folder, hr_file),
                    os.path.join(lr_folder, base_name + ".png"),
                )
            )

    return lr_hr_pairs


def save_meta_info(meta_info_file: str, lr_hr_pairs: list[tuple[str, str]]) -> None:
    with open(meta_info_file, "w") as f:
        for lr, hr in lr_hr_pairs:
            f.write(f"{lr}, {hr}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_folder", type=str, default="HR")
    parser.add_argument("--lr_folder", type=str, default="LR")
    parser.add_argument("--meta_info", type=str, default="meta_info_DIV2K_sub_pair.txt")
    args = parser.parse_args()

    lr_hr_pairs = []
    for lr_sub_folder in os.listdir(args.lr_folder):
        lr_sub_folder_path = os.path.join(args.lr_folder, lr_sub_folder)
        if os.path.isdir(lr_sub_folder_path):
            lr_hr_pairs.extend(get_lr_hr_pair(args.hr_folder, lr_sub_folder_path))

    save_meta_info(args.meta_info, lr_hr_pairs)
