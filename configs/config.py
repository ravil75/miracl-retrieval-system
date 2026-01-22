"""Конфигурация проекта"""

# URLs
BASE_URL = "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-ru/"
CORPUS_REPO = "miracl/miracl-corpus"

# Модели
E5_MODEL_NAME = "intfloat/multilingual-e5-base"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Пути к файлам
BPE_MODEL_PATH = "bpe_ru_25k.model"
DENSE_INDEX_PATH = "dense_e5.faiss"
TRAINED_INDEX_PATH = "trained_e5.faiss"

# Google Drive IDs
GDRIVE_DENSE_INDEX_ID = "1_z6Kup484-UKe4UxTTKRN2fzPmtYLsKn"
GDRIVE_BPE_MODEL_ID = "1kvmEcCUtGqj66q6NqXMcpJMCQv_Teqvf"

# Параметры данных
TARGET_DOCS = 500_000
SEED = 42

# Параметры обучения
BATCH_SIZE = 24
EPOCHS = 1
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 160
WARMUP_RATIO = 0.1

# Параметры поиска
HYBRID_ALPHA = 1.0
RERANK_TOP_N = 50
CANDIDATES = 100