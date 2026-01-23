"""Инференс для лучшей модели (Dense + Reranker)"""

from typing import List, Dict, Optional

from .models.dense import DenseRetrieverE5
from .models.reranker import Reranker
from configs.config import E5_MODEL_NAME, RERANKER_MODEL_NAME, RERANK_TOP_N


class FinalRetriever:
    """
    Retrieval система для голосового помощника Маруся
    
    Pipeline: Dense E5 → BGE Reranker
    """
    
    def __init__(
        self,
        dense_model: str = E5_MODEL_NAME,
        reranker_model: str = RERANKER_MODEL_NAME,
        rerank_top_n: int = RERANK_TOP_N
    ):
        self.dense = DenseRetrieverE5(dense_model)
        self.reranker = Reranker(reranker_model)
        self.rerank_top_n = rerank_top_n
        
        print("Retriever готов!")
    
    @property
    def doc_ids(self):
        return self.dense.doc_ids
    
    @property
    def passages(self):
        return self.dense.passages
    
    def load_index(self, index_path: str, doc_ids: List[str], passages: List[str]):
        """Загрузка FAISS индекса"""
        self.dense.load_index(index_path, doc_ids, passages)
    
    def build_index(self, doc_ids: List[str], passages: List[str], batch_size: int = 128):
        """Построение индекса"""
        self.dense.fit(doc_ids, passages, batch_size)
    
    def save_index(self, path: str):
        """Сохранение индекса"""
        self.dense.save_index(path)
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        use_reranker: bool = True
    ) -> List[Dict]:
        """
        Поиск по запросу
        
        Args:
            query: текст запроса
            top_k: сколько результатов вернуть
            use_reranker: использовать ли reranker
        """
        if use_reranker:
            candidates = self.dense.search(query, top_k=self.rerank_top_n)
            return self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            return self.dense.search(query, top_k=top_k)
    
    def answer(self, query: str) -> str:
        """Получить лучший ответ"""
        results = self.search(query, top_k=1)
        return results[0]['passage'] if results else "Ответ не найден"


def demo(retriever: FinalRetriever, queries: List[str] = None):
    """Демонстрация"""
    if queries is None:
        queries = [
            "Кто такой Юрий Гагарин?",
            "Столица России",
            "Кто написал Войну и мир?"
        ]
    
    print("="*60)
    print("🎯 Final RETRIEVER")
    print("="*60)
    
    for query in queries:
        print(f"\n📝 {query}")
        answer = retriever.answer(query)
        print(f"💬 {answer[:200]}..." if len(answer) > 200 else f"💬 {answer}")


def interactive(retriever: FinalRetriever):
    """Интерактивный режим"""
    print("="*60)
    print("🔍 ИНТЕРАКТИВНЫЙ ПОИСК (exit для выхода)")
    print("="*60)
    
    while True:
        query = input("\n📝 Вопрос: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query:
            continue
        
        results = retriever.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            passage = r['passage'][:120] + "..." if len(r['passage']) > 120 else r['passage']
            print(f"\n{i}. [{r['score']:.4f}] {passage}")