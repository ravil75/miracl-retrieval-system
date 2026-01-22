"""Hybrid Retriever: BM25 + Dense"""

from configs.config import HYBRID_ALPHA


class HybridRetriever:
    """Гибридный ретривер: BM25 + Dense"""
    
    def __init__(self, bm25_retriever, dense_retriever, alpha: float = HYBRID_ALPHA):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha
        self.doc_ids = dense_retriever.doc_ids
        self.passages = dense_retriever.passages
        
        print(f"Hybrid: {(1-alpha)*100:.0f}% BM25 + {alpha*100:.0f}% Dense")
    
    def search(self, query: str, top_k: int = 10, candidates: int = 100) -> list:
        bm25_results = self.bm25.search(query, top_k=candidates)
        dense_results = self.dense.search(query, top_k=candidates)
        
        # Нормализация BM25
        bm25_scores = [r['score'] for r in bm25_results]
        min_b, max_b = min(bm25_scores), max(bm25_scores)
        range_b = max_b - min_b + 1e-6
        
        all_docs = {}
        
        for r in bm25_results:
            norm_score = (r['score'] - min_b) / range_b
            all_docs[r['doc_id']] = {'bm25': norm_score, 'dense': 0.0, 'passage': r['passage']}
        
        for r in dense_results:
            if r['doc_id'] in all_docs:
                all_docs[r['doc_id']]['dense'] = r['score']
            else:
                all_docs[r['doc_id']] = {'bm25': 0.0, 'dense': r['score'], 'passage': r['passage']}
        
        results = []
        for doc_id, data in all_docs.items():
            combined = (1 - self.alpha) * data['bm25'] + self.alpha * data['dense']
            results.append({'doc_id': doc_id, 'score': combined, 'passage': data['passage']})
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]