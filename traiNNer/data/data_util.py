from concurrent.futures import ThreadPoolExecutor
from os import path as osp

import numpy as np

from traiNNer.utils import get_root_logger, scandir


def check_missing_paths(missing_from_paths: set[str], key: str, folder: str) -> None:
    if len(missing_from_paths) == 0:
        return

    missing_subset = sorted(missing_from_paths)[:10]
    raise ValueError(
        f"{len(missing_from_paths)} files are missing from {key}_paths ({folder}). The first few missing files are:\n"
        + "\n".join(missing_subset)
    )


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

    assert len(input_folders) == len(gt_folders), (
        "The lengths of input_folders and gt_folders must be the same."
    )

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

    assert len(input_folders) == len(gt_folders), (
        "The lengths of input_folders and gt_folders must be the same."
    )

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

    Matches files by stem (filename without extension), allowing mixed extensions
    like hr/1.jpg paired with lr/1.png.

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

        # Build stem -> full relative path mappings (stem = path without extension)
        gt_stem_to_name: dict[str, str] = {osp.splitext(f)[0]: f for f in gt_names}
        input_stem_to_name: dict[str, str] = {
            osp.splitext(f)[0]: f for f in input_names
        }

        gt_stems = set(gt_stem_to_name.keys())
        input_stems = set(input_stem_to_name.keys())

        if filename_tmpl == "{}":
            missing_from_gt = input_stems - gt_stems
            missing_from_input = gt_stems - input_stems

            check_missing_paths(missing_from_gt, gt_key, gt_folder)
            check_missing_paths(missing_from_input, input_key, input_folder)

            assert len(input_stems) == len(gt_stems), (
                f"{input_key} and {gt_key} datasets have different number of images: "
                f"{len(input_stems)}, {len(gt_stems)}."
            )

            for stem in sorted(gt_stems):
                paired_paths.append(
                    {
                        f"{input_key}_path": osp.join(
                            input_folder, input_stem_to_name[stem]
                        ),
                        f"{gt_key}_path": osp.join(gt_folder, gt_stem_to_name[stem]),
                    }
                )
        else:
            # Template case: input filename stem is derived from gt basename via template
            for gt_name in sorted(gt_names):
                gt_dir = osp.dirname(gt_name)
                gt_basename = osp.splitext(osp.basename(gt_name))[0]

                # Build expected input stem using template
                templated_basename = filename_tmpl.format(gt_basename)
                input_stem = (
                    osp.join(gt_dir, templated_basename)
                    if gt_dir
                    else templated_basename
                )

                if input_stem in input_stem_to_name:
                    paired_paths.append(
                        {
                            f"{input_key}_path": osp.join(
                                input_folder, input_stem_to_name[input_stem]
                            ),
                            f"{gt_key}_path": osp.join(gt_folder, gt_name),
                        }
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
