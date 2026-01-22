"""Скачивание FAISS индекса"""

import gdown
from configs.config import GDRIVE_INDEX_ID, DENSE_INDEX_PATH


def download_index(output_path: str = DENSE_INDEX_PATH):
    url = f"https://drive.google.com/uc?id={GDRIVE_INDEX_ID}&export=download"
    gdown.download(url, output_path, quiet=False)
    print(f"Индекс скачан: {output_path}")


if __name__ == "__main__":
    download_index()