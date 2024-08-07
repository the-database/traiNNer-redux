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


def paired_paths_from_lmdb(folders: list[str], keys: list[str]) -> list[dict[str, str]]:
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    ::

        lq.lmdb
        ├── data.mdb
        ├── lock.mdb
        ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert (
        len(keys) == 2
    ), f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith(".lmdb") and gt_folder.endswith(".lmdb")):
        raise ValueError(
            f"{input_key} folder and {gt_key} folder should both in lmdb "
            f"formats. But received {input_key}: {input_folder}; "
            f"{gt_key}: {gt_folder}"
        )
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, "meta_info.txt")) as fin:
        input_lmdb_keys = [line.split(".")[0] for line in fin]
    with open(osp.join(gt_folder, "meta_info.txt")) as fin:
        gt_lmdb_keys = [line.split(".")[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f"Keys in {input_key}_folder and {gt_key}_folder are different."
        )
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append({f"{input_key}_path": lmdb_key, f"{gt_key}_path": lmdb_key})
        return paths


def paired_paths_from_meta_info_file(
    folders: list[str], keys: list[str], meta_info_file: str, filename_tmpl: str
) -> list[dict[str, str]]:
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert (
        len(keys) == 2
    ), f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file) as fin:
        gt_names = [line.strip().split(" ")[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f"{filename_tmpl.format(basename)}{ext}"
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append({f"{input_key}_path": input_path, f"{gt_key}_path": gt_path})
    return paths


def paired_paths_from_folder(
    folders: tuple[str, str], keys: tuple[str, str], filename_tmpl: str
) -> list[dict[str, str]]:
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """

    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if input_folder == gt_folder:
        logger = get_root_logger()
        logger.warning(
            "%s and %s datasets have the same path, this may be unintentional: %s",
            input_key,
            gt_key,
            gt_folder,
        )

    gt_names = list(scandir(gt_folder, recursive=True))
    gt_paths = [(f"{gt_key}_path", osp.join(gt_folder, f)) for f in gt_names]

    input_names = list(scandir(input_folder, recursive=True))

    assert len(input_names) == len(gt_names), (
        f"{input_key} and {gt_key} datasets have different number of images: "
        f"{len(input_names)}, {len(gt_names)}."
    )

    input_set = set(input_names)
    gt_set = set(gt_names)
    missing_from_gt_paths = input_set - gt_set
    missing_from_input_paths = gt_set - input_set

    check_missing_paths(missing_from_gt_paths, gt_key, gt_folder)
    check_missing_paths(missing_from_input_paths, input_key, input_folder)

    if filename_tmpl == "{}":
        input_paths = [
            (f"{input_key}_path", osp.join(input_folder, f)) for f in gt_names
        ]
    else:
        gt_basename_ext = [osp.splitext(osp.basename(gt_name)) for gt_name in gt_names]
        input_paths = [
            (
                f"{input_key}_path",
                osp.join(input_folder, f"{filename_tmpl.format(basename)}{ext}"),
            )
            for basename, ext in gt_basename_ext
        ]

    return [dict([a, b]) for a, b in zip(input_paths, gt_paths, strict=False)]


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


def paths_from_lmdb(folder: str) -> list[str]:
    """Generate paths from lmdb.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    if not folder.endswith(".lmdb"):
        raise ValueError(f"Folder {folder}folder should in lmdb format.")
    with open(osp.join(folder, "meta_info.txt")) as fin:
        paths = [line.split(".")[0] for line in fin]
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
