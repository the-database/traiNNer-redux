import os
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from chainner_ext import ResizeFilter, resize
from cv2.typing import MatLike
from PIL import Image
from tqdm import tqdm

DRY_RUN = False  # Set this to False when you want to actually copy the files


def chroma_subsampling_420(image: MatLike, interpolation: int) -> MatLike:
    """Perform chroma subsampling 4:2:0 using bicubic interpolation."""
    # Convert the image to YUV color space
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Get the Y, U, and V channels
    y = yuv[:, :, 0]
    u = yuv[:, :, 1]
    v = yuv[:, :, 2]

    # downsample u and v channels to half resolution
    u_downsampled = cv2.resize(
        u, (u.shape[1] // 2, u.shape[0] // 2), interpolation=interpolation
    )
    v_downsampled = cv2.resize(
        v, (v.shape[1] // 2, v.shape[0] // 2), interpolation=interpolation
    )

    # create a new yuv420 image
    h, w = y.shape
    yuv420 = np.zeros((h, w, 3), dtype=np.uint8)
    yuv420[:, :, 0] = y  # Y channel

    # Place the downsampled U and V channels in the proper locations
    yuv420[0:h:2, 0:w:2, 1] = u_downsampled  # u channel (downsampled)
    yuv420[0:h:2, 0:w:2, 2] = v_downsampled  # v channel (downsampled)

    # upsample u and v channels back to the original size (for visualization)
    yuv420[:, :, 1] = cv2.resize(u_downsampled, (w, h), interpolation=interpolation)
    yuv420[:, :, 2] = cv2.resize(v_downsampled, (w, h), interpolation=interpolation)

    # Convert back to RGB
    return cv2.cvtColor(yuv420, cv2.COLOR_YCrCb2BGR)


def split_file_path(path: Path | str) -> tuple[Path, str, str]:
    """
    Returns the base directory, file name, and extension of the given file path.
    """
    base, ext = os.path.splitext(path)
    dirname, basename = os.path.split(base)
    return Path(dirname), basename, ext


def _read_cv_from_path(path: str) -> MatLike:
    img = None
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        pass

    if img is None:
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        except Exception as e:
            raise RuntimeError(
                f'Error reading image image from path "{path}". Image may be corrupt.'
            ) from e

    if img is None:  # type: ignore
        raise RuntimeError(
            f'Error reading image image from path "{path}". Image may be corrupt.'
        )

    return img


def custom_resize(image: MatLike, size_ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    pil_image = Image.fromarray(image)
    # print(w, h, new_size)

    new_image = np.array(pil_image)
    new_image = new_image.astype(np.float32) / 255.0
    new_image = resize(
        new_image,
        (round(w * size_ratio), round(h * size_ratio)),
        ResizeFilter.CubicCatrom,
        False,
    )
    new_image = (new_image * 255).astype(np.uint8)

    pil_image = Image.fromarray(new_image)
    return np.array(pil_image)


def downscale_image(args: tuple[str, str, float]) -> None:
    input_path, output_path, size_ratio = args
    if not DRY_RUN:
        image = _read_cv_from_path(input_path)
        image = custom_resize(image, size_ratio)

        if random.random() < 0.5:
            if random.random() < 0.5:
                image = chroma_subsampling_420(
                    image, interpolation=cv2.INTER_NEAREST_EXACT
                )
            else:
                image = chroma_subsampling_420(
                    image, interpolation=cv2.INTER_LINEAR_EXACT
                )
        else:
            pass

        cv2.imwrite(output_path, image)
    else:
        print(output_path)


def process_images(
    input_directory: str, output_directory: str, size_ratio: float
) -> None:
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image_paths = []

    for root, _, files in os.walk(input_directory):
        image_files = []
        for file in files:
            image_files.append(file)

        selected_files = image_files

        for file in selected_files:
            input_image_path = os.path.join(root, file)
            output_subdirectory = os.path.relpath(root, input_directory)
            output_subdirectory_path = os.path.join(
                output_directory, output_subdirectory
            )
            output_image_path = os.path.join(output_subdirectory_path, file)

            if not DRY_RUN and not os.path.exists(output_subdirectory_path):
                os.makedirs(output_subdirectory_path)

            image_paths.append((input_image_path, output_image_path, size_ratio))

    with Pool(processes=8) as pool, tqdm(total=len(image_paths)) as pbar:
        for _ in pool.imap_unordered(downscale_image, image_paths):
            pbar.update(1)


if __name__ == "__main__":
    scale = 4
    scale_str = f"x{scale}"
    size_ratio = 1 / scale

    input_directory = r"C:\path\to\input\dir"
    output_directory = r"C:\path\to\output\dir"
    process_images(input_directory, output_directory, size_ratio)
