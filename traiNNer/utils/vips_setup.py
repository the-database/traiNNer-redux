import os
import platform
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

from traiNNer.utils.logger import get_root_logger


def setup_vips() -> None:
    logger = get_root_logger()
    cache_dir = Path.home() / ".cache"
    vips_dir = cache_dir / "vips"
    vips_bin = vips_dir / "vips-dev-8.16" / "bin"

    os.environ["PATH"] = os.pathsep.join((str(vips_bin), os.environ["PATH"]))

    system = platform.system()

    if system == "Linux":
        try:
            subprocess.run(["vips", "--version"], check=True, stdout=subprocess.PIPE)
            print("VIPS is already installed and available in PATH.")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(
                "libvips is not installed. Please install libvibs using the following command:"
                "  sudo apt install libvips-dev --no-install-recommends"
            )

            sys.exit(1)
    elif system == "Windows":
        vips_url = "https://github.com/libvips/build-win64-mxe/releases/download/v8.16.0/vips-dev-w64-all-8.16.0.zip"
        zip_path = cache_dir / "vips.zip"

        if not vips_bin.is_dir():
            logger.info("VIPS not found, downloading and installing for Windows...")
            os.makedirs(cache_dir, exist_ok=True)
            urllib.request.urlretrieve(vips_url, zip_path)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(vips_dir)

            os.remove(zip_path)
            logger.info("VIPS installed at %s", vips_dir)

        if str(vips_bin) not in os.environ["PATH"]:
            os.environ["PATH"] = os.pathsep.join((str(vips_bin), os.environ["PATH"]))


def initialize_vips() -> ModuleType:
    setup_vips()
    try:
        import pyvips

        return pyvips
    except ModuleNotFoundError:
        # hack
        return SimpleNamespace(Image="")  # type: ignore
