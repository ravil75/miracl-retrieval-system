"""Скачивание необходимых файлов с Google Drive"""

import os
import gdown

from configs.config import (
    GDRIVE_DENSE_INDEX_ID, 
    GDRIVE_BPE_MODEL_ID,
    DENSE_INDEX_PATH,
    BPE_MODEL_PATH
)


def download_from_gdrive(file_id: str, output_path: str, description: str = ""):
    """Скачивание файла с Google Drive"""
    if os.path.exists(output_path):
        print(f"✓ {description} уже существует: {output_path}")
        return
    
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    print(f"Скачивание {description}...")
    gdown.download(url, output_path, quiet=False)
    print(f"✓ Скачан: {output_path}")


def download_dense_index(output_path: str = DENSE_INDEX_PATH):
    """Скачивание FAISS индекса для Dense Retriever"""
    download_from_gdrive(
        GDRIVE_DENSE_INDEX_ID, 
        output_path, 
        "FAISS индекс (Dense E5)"
    )


def download_bpe_model(output_path: str = BPE_MODEL_PATH):
    """Скачивание BPE токенизатора для BM25"""
    download_from_gdrive(
        GDRIVE_BPE_MODEL_ID, 
        output_path, 
        "BPE токенизатор"
    )


def download_all():
    """Скачивание всех необходимых файлов"""
    print("="*50)
    print("📥 СКАЧИВАНИЕ ФАЙЛОВ")
    print("="*50)
    
    download_bpe_model()
    download_dense_index()
    
    print("\n✓ Все файлы скачаны!")


if __name__ == "__main__":
    download_all()