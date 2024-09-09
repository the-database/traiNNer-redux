import cv2
from cv2.typing import MatLike
from traiNNer.metrics.psnr_ssim import calculate_ssim


def open_image(file_path: str) -> MatLike:
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main() -> None:
    img1 = open_image(r"./img1.png")
    img2 = open_image(r"./img2.png")

    ssim_value = calculate_ssim(img1, img2, crop_border=4)

    print(f"SSIM: {ssim_value}")


if __name__ == "__main__":
    main()
