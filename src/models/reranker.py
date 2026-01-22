"""Reranker и Full Pipeline"""

import torch
from sentence_transformers import CrossEncoder

from configs.config import RERANKER_MODEL_NAME, RERANK_TOP_N


class Reranker:
    """Cross-Encoder Reranker"""
    
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        print(f"Загрузка Reranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512, trust_remote_code=True)
        
        if torch.cuda.is_available():
            self.model.model.to('cuda')
            self.model.model.half()
    
    def rerank(self, query: str, candidates: list, top_k: int = 10) -> list:
        if not candidates:
            return []
        
        pairs = [(query, c['passage']) for c in candidates]
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        
        reranked = [{'doc_id': c['doc_id'], 'score': float(s), 'passage': c['passage']} 
                    for c, s in zip(candidates, scores)]
        reranked.sort(key=lambda x: x['score'], reverse=True)
        return reranked[:top_k]


class FullPipeline:
    """Hybrid + Reranker Pipeline"""
    
    def __init__(self, hybrid_retriever, reranker, rerank_top_n: int = RERANK_TOP_N):
        self.hybrid = hybrid_retriever
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        self.doc_ids = hybrid_retriever.doc_ids
        self.passages = hybrid_retriever.passages
        
        print(f"Pipeline: Hybrid → Rerank top-{rerank_top_n}")
    
    def search(self, query: str, top_k: int = 10) -> list:
        candidates = self.hybrid.search(query, top_k=self.rerank_top_n)
        return self.reranker.rerank(query, candidates, top_k=top_k)