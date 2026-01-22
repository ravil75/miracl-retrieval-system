"""Метрики оценки"""

import numpy as np
from collections import defaultdict
from tqdm.notebook import tqdm


def evaluate(retriever, queries: dict, qrels: dict, ks: list = [1, 5, 10, 20, 100]) -> dict:
    """Оценка качества ретривера"""
    metrics = defaultdict(list)
    indexed = set(retriever.doc_ids)
    
    for qid, text in tqdm(queries.items(), desc="Оценка"):
        if qid not in qrels:
            continue
        
        relevant = [d for d in qrels[qid]['positive'] if d in indexed]
        if not relevant:
            continue
        
        results = retriever.search(text, top_k=max(ks))
        retrieved = [r['doc_id'] for r in results]
        
        # Recall@K
        for k in ks:
            hit = len(set(retrieved[:k]) & set(relevant))
            metrics[f'Recall@{k}'].append(hit / len(relevant))
        
        # MRR
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in set(relevant):
                metrics['MRR'].append(1.0 / rank)
                break
        else:
            metrics['MRR'].append(0.0)
        
        # NDCG@10
        dcg = sum(1.0/np.log2(i+2) for i, d in enumerate(retrieved[:10]) if d in set(relevant))
        idcg = sum(1.0/np.log2(i+2) for i in range(min(10, len(relevant))))
        metrics['NDCG@10'].append(dcg / idcg if idcg > 0 else 0)
    
    return {k: np.mean(v) for k, v in metrics.items()}


def print_comparison(baseline: dict, current: dict, name: str = "Current"):
    """Вывод сравнения метрик"""
    print(f"\n{'Метрика':<12} {'Baseline':<12} {name:<12} {'Δ':<10}")
    print("-" * 48)
    for m in ['MRR', 'NDCG@10', 'Recall@10']:
        b = baseline.get(m, 0)
        c = current.get(m, 0)
        d = c - b
        sign = "+" if d > 0 else ""
        status = "✓" if d > 0 else ""
        print(f"{m:<12} {b:<12.4f} {c:<12.4f} {sign}{d:.4f} {status}")