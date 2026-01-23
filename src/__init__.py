"""MIRACL Retrieval System"""

from .data.loader import load_queries, load_qrels, load_corpus, get_required_doc_ids, load_all_data
from .data.dataset import E5Dataset, E5PairDataset

from .models.bm25 import BM25_BPE
from .models.dense import DenseRetrieverE5
from .models.hybrid import HybridRetriever
from .models.reranker import Reranker, FullPipeline

from .evaluation.metrics import evaluate, print_comparison

from .training.trainer import train_model, prepare_training_data

# Функции скачивания
from scripts.download_assets import download_all, download_bpe_model, download_dense_index

from .inference import FinalRetriever, demo, interactive