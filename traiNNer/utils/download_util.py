import math
import os
from http.client import HTTPResponse
from http.cookiejar import CookieJar
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener

from torch.hub import download_url_to_file, get_dir
from tqdm import tqdm

from traiNNer.utils.misc import sizeof_fmt


def download_file_from_google_drive(file_id: str, save_path: str) -> None:
    """Download files from google drive.

    Reference: https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive

    Args:
        file_id (str): File id.
        save_path (str): Save path.
    """

    cookies = CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookies))
    url = "https://docs.google.com/uc?export=download"
    params = {"id": file_id}

    response = opener.open(f"{url}&{urlencode(params)}")
    token = get_confirm_token(cookies)
    if token:
        params["confirm"] = token
        response.close()
        response = opener.open(f"{url}&{urlencode(params)}")

    # get file size
    with opener.open(
        Request(f"{url}&{urlencode(params)}", headers={"Range": "bytes=0-2"})
    ) as response_file_size:
        content_range = response_file_size.headers.get("Content-Range")
    file_size = int(content_range.split("/")[1]) if content_range else None

    with response:
        save_response_content(response, save_path, file_size)


def get_confirm_token(cookies: CookieJar) -> str | None:
    for cookie in cookies:
        if cookie.name.startswith("download_warning"):
            return cookie.value
    return None


def save_response_content(
    response: HTTPResponse,
    destination: str,
    file_size: int | None = None,
    chunk_size: int = 32768,
) -> None:
    if file_size is not None:
        pbar = tqdm(total=math.ceil(file_size / chunk_size), unit="chunk")

        readable_file_size = sizeof_fmt(file_size)
    else:
        pbar = None
        readable_file_size = None

    with open(destination, "wb") as f:
        downloaded_size = 0
        while chunk := response.read(chunk_size):
            downloaded_size += chunk_size
            if pbar is not None and readable_file_size is not None:
                pbar.update(1)
                pbar.set_description(
                    f"Download {sizeof_fmt(downloaded_size)} / {readable_file_size}"
                )
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
        if pbar is not None:
            pbar.close()


def load_file_from_url(
    url: str,
    model_dir: str | None = None,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Load file form http url, will download models if necessary.

    Reference: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file
