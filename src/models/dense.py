"""Dense Retriever с E5"""

import gc
import time
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm

from configs.config import E5_MODEL_NAME


class DenseRetrieverE5:
    """Dense Retriever с multilingual-e5"""
    
    def __init__(self, model_name: str = E5_MODEL_NAME):
        print(f"Загрузка модели: {model_name}")
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cuda':
            self.model = self.model.to('cuda')
            self.model.half()
        
        self.index = None
        self.doc_ids = None
        self.passages = None
        self.dimension = None
    
    def _prepare_passage(self, text: str) -> str:
        return f"passage: {text}"
    
    def _prepare_query(self, text: str) -> str:
        return f"query: {text}"
    
    def fit(self, doc_ids: list, passages: list, batch_size: int = 128):
        """Построение индекса"""
        print("Построение индекса...")
        
        self.doc_ids = doc_ids
        self.passages = passages
        n_docs = len(passages)
        
        # Размерность
        with torch.no_grad():
            sample = self.model.encode(
                [self._prepare_passage(passages[0])],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        self.dimension = sample.shape[1]
        
        # FAISS индекс
        self.index = faiss.IndexFlatIP(self.dimension)
        
        start_time = time.time()
        
        for start_idx in tqdm(range(0, n_docs, batch_size), desc="Индексация"):
            end_idx = min(start_idx + batch_size, n_docs)
            batch = [self._prepare_passage(p) for p in passages[start_idx:end_idx]]
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            
            self.index.add(embeddings.astype('float32'))
            
            if (start_idx // batch_size) % 50 == 0:
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        print(f"Индекс построен: {n_docs:,} документов за {elapsed/60:.1f} мин")
    
    def search(self, query: str, top_k: int = 10) -> list:
        prepared_query = self._prepare_query(query)
        
        with torch.no_grad():
            query_emb = self.model.encode(
                [prepared_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')
        
        scores, indices = self.index.search(query_emb, top_k)
        
        return [{
            'doc_id': self.doc_ids[idx],
            'score': float(score),
            'passage': self.passages[idx]
        } for idx, score in zip(indices[0], scores[0])]
    
    def save_index(self, path: str):
        faiss.write_index(self.index, path)
        print(f"Индекс сохранён: {path}")
    
    def load_index(self, path: str, doc_ids: list, passages: list):
        self.index = faiss.read_index(path)
        self.doc_ids = doc_ids
        self.passages = passages
        self.dimension = self.index.d
        print(f"Индекс загружен: {self.index.ntotal:,} документов")