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

DRY_RUN = False  # Set this to False to write output files


def pad_image_for_chroma(image: MatLike, scale: int) -> tuple[MatLike, int, int]:
    """Pad high-res image bottom/right to multiple of (scale * 2) so that low-res dims will be even."""
    h, w = image.shape[:2]
    align = scale * 2
    pad_bottom = (-h) % align
    pad_right = (-w) % align
    if pad_bottom or pad_right:
        image = cv2.copyMakeBorder(
            image, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_REFLECT
        )
    return image, pad_bottom, pad_right


def chroma_subsampling_420(image: MatLike, interpolation: int) -> MatLike:
    """Perform chroma subsampling 4:2:0 using bicubic interpolation."""
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y = yuv[:, :, 0]
    u = yuv[:, :, 1]
    v = yuv[:, :, 2]

    # downsample u/v to half resolution
    u_down = cv2.resize(
        u, (u.shape[1] // 2, u.shape[0] // 2), interpolation=interpolation
    )
    v_down = cv2.resize(
        v, (v.shape[1] // 2, v.shape[0] // 2), interpolation=interpolation
    )

    h, w = y.shape
    yuv420 = np.zeros((h, w, 3), dtype=np.uint8)
    yuv420[:, :, 0] = y
    yuv420[0:h:2, 0:w:2, 1] = u_down
    yuv420[0:h:2, 0:w:2, 2] = v_down
    # upsample back
    yuv420[:, :, 1] = cv2.resize(u_down, (w, h), interpolation=interpolation)
    yuv420[:, :, 2] = cv2.resize(v_down, (w, h), interpolation=interpolation)

    return cv2.cvtColor(yuv420, cv2.COLOR_YCrCb2BGR)


def _read_cv_from_path(path: str) -> MatLike:
    img = None
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        pass
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'Error reading image from "{path}".')
    return img


def custom_resize(image: MatLike, size_ratio: float) -> np.ndarray:
    pil_img = Image.fromarray(image)
    arr = np.array(pil_img).astype(np.float32) / 255.0
    new_w = round(image.shape[1] * size_ratio)
    new_h = round(image.shape[0] * size_ratio)
    resized = resize(arr, (new_w, new_h), ResizeFilter.CubicCatrom, False)
    return (resized * 255).astype(np.uint8)


def downscale_image(args: tuple[str, str, float]) -> None:
    input_path, output_path, size_ratio = args
    if DRY_RUN:
        print(output_path)
        return

    image = _read_cv_from_path(input_path)
    h_orig, w_orig = image.shape[:2]
    scale = round(1 / size_ratio)

    # 1) Pad high-res image for chroma subsampling
    image_padded, _pad_hr_b, _pad_hr_r = pad_image_for_chroma(image, scale)

    # 2) Resize to low-res
    lr_padded = custom_resize(image_padded, size_ratio)

    # 3) Chroma subsampling
    lr_sub = chroma_subsampling_420(lr_padded, interpolation=cv2.INTER_LINEAR_EXACT)

    # 4) Crop back to original low-res dimensions
    orig_lr_h = round(h_orig * size_ratio)
    orig_lr_w = round(w_orig * size_ratio)
    lr_h, lr_w = lr_sub.shape[:2]
    crop_h = min(orig_lr_h, lr_h)
    crop_w = min(orig_lr_w, lr_w)
    lr_final = lr_sub[:crop_h, :crop_w]

    # 5) Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, lr_final)


def process_images(
    input_directory: str, output_directory: str, size_ratio: float
) -> None:
    os.makedirs(output_directory, exist_ok=True)
    tasks = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            in_p = os.path.join(root, file)
            rel = os.path.relpath(root, input_directory)
            out_sub = os.path.join(output_directory, rel)
            out_p = os.path.join(out_sub, file)
            tasks.append((in_p, out_p, size_ratio))

    with Pool(processes=8) as pool, tqdm(total=len(tasks)) as pbar:
        for _ in pool.imap_unordered(downscale_image, tasks):
            pbar.update(1)


if __name__ == "__main__":
    scale = 4
    size_ratio = 1 / scale

    input_directory = r"C:\path\to\input\dir"
    output_directory = r"C:\path\to\output\dir"
    process_images(input_directory, output_directory, size_ratio)
