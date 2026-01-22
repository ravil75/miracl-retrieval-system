"""BM25 с BPE токенизацией"""

import gc
import numpy as np
import sentencepiece as spm
from rank_bm25 import BM25Okapi
from tqdm.notebook import tqdm


class BM25_BPE:
    """BM25 с SentencePiece BPE токенизацией"""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.bm25 = None
        self.doc_ids = None
        self.passages = None
        self.min_token_len = 2

    def tokenize(self, text: str) -> list:
        tokens = self.sp.encode_as_pieces(text)
        filtered = []
        for token in tokens:
            clean_token = token.replace("▁", "")
            if len(clean_token) < self.min_token_len:
                continue
            if clean_token in '.,!?;:—–-()[]{}«»"\'':
                continue
            filtered.append(token)
        return filtered

    def fit(self, doc_ids: list, passages: list):
        """Построение BM25 индекса"""
        self.doc_ids = doc_ids
        self.passages = passages

        tokenized_corpus = [self.tokenize(p) for p in tqdm(passages, desc="Токенизация")]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        del tokenized_corpus
        gc.collect()
        
        print(f"BM25 индекс: {len(doc_ids):,} документов")

    def search(self, query: str, top_k: int = 10) -> list:
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [{
            'doc_id': self.doc_ids[idx],
            'score': float(scores[idx]),
            'passage': self.passages[idx]
        } for idx in top_indices]