from concurrent.futures import ThreadPoolExecutor
from os import path as osp

import cv2
import numpy as np
import torch
from torch import Tensor

from traiNNer.data.transforms import mod_crop
from traiNNer.utils import get_root_logger, imgs2tensors, scandir


def check_missing_paths(missing_from_paths: set[str], key: str, folder: str) -> None:
    if len(missing_from_paths) == 0:
        return

    missing_subset = sorted(missing_from_paths)[:10]
    raise ValueError(
        f"{len(missing_from_paths)} files are missing from {key}_paths ({folder}). The first few missing files are:\n"
        + "\n".join(missing_subset)
    )


def read_img_seq(
    path: str | list[str],
    require_mod_crop: bool = False,
    scale: int = 1,
    return_imgname: bool = False,
) -> Tensor | tuple[Tensor, list[str]]:
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname(bool): Whether return image names. Default False.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
        list[str]: Returned image name list.
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(scandir(path, full_path=True))
    imgs = [cv2.imread(v).astype(np.float32) / 255.0 for v in img_paths]

    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = imgs2tensors(imgs, bgr2rgb=True, float32=True)
    assert isinstance(imgs, list)
    imgs = torch.stack(imgs, dim=0)

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return imgs, imgnames
    else:
        return imgs


def generate_frame_indices(
    crt_idx: int, max_frame_num: int, num_frames: int, padding: str = "reflection"
) -> list[int]:
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, "num_frames should be an odd number."
    assert padding in (
        "replicate",
        "reflection",
        "reflection_circle",
        "circle",
    ), f"Wrong padding mode: {padding}."

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == "replicate":
                pad_idx = 0
            elif padding == "reflection":
                pad_idx = -i
            elif padding == "reflection_circle":
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == "replicate":
                pad_idx = max_frame_num
            elif padding == "reflection":
                pad_idx = max_frame_num * 2 - i
            elif padding == "reflection_circle":
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def paired_paths_from_lmdb(
    folders: tuple[list[str], list[str]], keys: tuple[str, str]
) -> list[dict[str, str]]:
    """Generate paired paths from LMDB files for multiple folder pairs, optimized for large datasets.

    Args:
        folders (tuple[list[str], list[str]]): Two lists of folder paths.
            The first list is for input folders, and the second is for GT folders.
        keys (tuple[list[str], list[str]]): Two lists of keys identifying folders.
            The first list corresponds to input folders, and the second to GT folders.
            Note that these keys are different from LMDB keys.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing paired LMDB paths.
    """
    input_folders, gt_folders = folders
    input_key, gt_key = keys
    paired_paths = []

    assert len(input_folders) == len(
        gt_folders
    ), "The lengths of input_folders and gt_folders must be the same."

    def process_folder_pair(
        input_folder: str, gt_folder: str, input_key: str, gt_key: str
    ) -> list[dict[str, str]]:
        if not (input_folder.endswith(".lmdb") and gt_folder.endswith(".lmdb")):
            raise ValueError(
                f"{input_key} folder and {gt_key} folder should both be in LMDB formats. "
                f"Received {input_key}: {input_folder}; {gt_key}: {gt_folder}"
            )

        input_meta_file = osp.join(input_folder, "meta_info.txt")
        gt_meta_file = osp.join(gt_folder, "meta_info.txt")

        with open(input_meta_file) as fin_input, open(gt_meta_file) as fin_gt:
            input_lmdb_keys = {line.split(".")[0] for line in fin_input}
            gt_lmdb_keys = {line.split(".")[0] for line in fin_gt}

        if input_lmdb_keys != gt_lmdb_keys:
            raise ValueError(
                f"Keys in {input_key}_folder and {gt_key}_folder are different."
            )

        return [
            {f"{input_key}_path": lmdb_key, f"{gt_key}_path": lmdb_key}
            for lmdb_key in sorted(input_lmdb_keys)
        ]

    with ThreadPoolExecutor() as executor:
        future_paths = executor.map(
            process_folder_pair,
            input_folders,
            gt_folders,
            input_key,
            gt_key,
        )

        for paths in future_paths:
            paired_paths.extend(paths)

    return paired_paths


def paired_paths_from_meta_info_file(
    folders: tuple[list[str], list[str]],
    keys: tuple[str, str],
    meta_info_file: str,
    filename_tmpl: str,
) -> list[dict[str, str]]:
    """Generate paired paths from a meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by white space.

    Example of a meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (tuple[list[str], list[str]]): Two lists of folder paths.
            The first list is for input folders, and the second is for GT folders.
        keys (tuple[list[str], list[str]]): Two lists of keys identifying folders.
            The first list corresponds to input folders, and the second to GT folders.
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folders.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing paired paths.
    """
    input_folders, gt_folders = folders
    input_key, gt_key = keys
    paired_paths = []

    assert len(input_folders) == len(
        gt_folders
    ), "The lengths of input_folders and gt_folders must be the same."

    with open(meta_info_file) as fin:
        gt_names = [line.strip().split(" ")[0] for line in fin]

    def process_paths(
        input_folder: str, gt_folder: str, input_key: str, gt_key: str
    ) -> list[dict[str, str]]:
        paths = []
        for gt_name in gt_names:
            basename, ext = osp.splitext(osp.basename(gt_name))
            input_name = f"{filename_tmpl.format(basename)}{ext}"
            input_path = osp.join(input_folder, input_name)
            gt_path = osp.join(gt_folder, gt_name)
            paths.append({f"{input_key}_path": input_path, f"{gt_key}_path": gt_path})
        return paths

    with ThreadPoolExecutor() as executor:
        results = executor.map(
            process_paths, input_folders, gt_folders, input_key, gt_key
        )

    for res in results:
        paired_paths.extend(res)

    return paired_paths


def paired_paths_from_folder(
    folders: tuple[list[str], list[str]],
    keys: tuple[str, str],
    filename_tmpl: str,
) -> list[dict[str, str]]:
    """Generate paired paths from multiple folders.

    Args:
        folders (tuple[list[str], list[str]]): Two lists of folder paths.
            The first list is for input folders, and the second is for GT folders.
        keys (tuple[list[str], list[str]]): Two lists of keys identifying folders.
            The first list corresponds to input folders, and the second to GT folders.
        filename_tmpl (str): Template for each filename. Note that the template
            excludes the file extension. Usually, the filename_tmpl is for files
            in the input folders.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing paired paths.
    """
    input_folders, gt_folders = folders
    input_key, gt_key = keys
    paired_paths = []

    def scan_folder(folder: str) -> list[str]:
        return list(scandir(folder, recursive=True))

    with ThreadPoolExecutor() as executor:
        folder_scans = list(executor.map(scan_folder, input_folders + gt_folders))

    input_scans = folder_scans[: len(input_folders)]
    gt_scans = folder_scans[len(input_folders) :]

    for input_folder, gt_folder, input_names, gt_names in zip(
        input_folders,
        gt_folders,
        input_scans,
        gt_scans,
        strict=False,
    ):
        if input_folder == gt_folder:
            logger = get_root_logger()
            logger.warning(
                "%s and %s datasets have the same path, this may be unintentional: %s",
                input_key,
                gt_key,
                gt_folder,
            )

        gt_paths = [(f"{gt_key}_path", osp.join(gt_folder, f)) for f in gt_names]
        input_set = set(input_names)
        gt_set = set(gt_names)

        assert len(input_names) == len(gt_names), (
            f"{input_key} and {gt_key} datasets have different number of images: "
            f"{len(input_names)}, {len(gt_names)}."
        )

        missing_from_gt_paths = input_set - gt_set
        missing_from_input_paths = gt_set - input_set

        if filename_tmpl == "{}":
            check_missing_paths(missing_from_gt_paths, gt_key, gt_folder)
            check_missing_paths(missing_from_input_paths, input_key, input_folder)

            input_paths = [
                (f"{input_key}_path", osp.join(input_folder, f)) for f in gt_names
            ]
        else:
            gt_basename_ext = [
                osp.splitext(osp.basename(gt_name)) for gt_name in gt_names
            ]
            input_paths = [
                (
                    f"{input_key}_path",
                    osp.join(input_folder, f"{filename_tmpl.format(basename)}{ext}"),
                )
                for basename, ext in gt_basename_ext
            ]

        paired_paths.extend(
            [dict([a, b]) for a, b in zip(input_paths, gt_paths, strict=False)]
        )

    return paired_paths


def paths_from_folder(folder: str) -> list[str]:
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def paths_from_lmdb(folders: list[str]) -> list[str]:
    """Generate flattened list of paths from multiple LMDB folders.

    Args:
        folders (list[str]): List of LMDB folder paths.

    Returns:
        list[str]: Flattened list of paths from all LMDB folders.
    """

    def process_folder(folder: str) -> list[str]:
        if not folder.endswith(".lmdb"):
            raise ValueError(f"Folder {folder} should be in LMDB format.")

        meta_info_file = osp.join(folder, "meta_info.txt")
        with open(meta_info_file) as fin:
            return [line.split(".")[0] for line in fin]

    with ThreadPoolExecutor() as executor:
        paths = [
            path
            for folder_paths in executor.map(process_folder, folders)
            for path in folder_paths
        ]

    return paths


def generate_gaussian_kernel(kernel_size: int = 13, sigma: float = 1.6) -> np.ndarray:
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters

    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)  # type: ignore
