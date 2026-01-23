#  Retrieval System для русского языка

Система информационного поиска на основе датасета MIRACL. Находит релевантные документы по текстовому запросу.

## Что это

Это учебный проект по созданию retrieval-системы. Цель — научиться строить пайплайны информационного поиска: от простого BM25 до нейросетевых подходов с reranking.

**Пример:**

Запрос: "Кто такой Юрий Гагарин?"

Ответ: "Юрий Алексеевич Гагарин — советский лётчик-космонавт,
Герой Советского Союза. 12 апреля 1961 года стал первым
человеком в мировой истории, совершившим полёт в космос."


## Результаты

| Метод | MRR | Recall@10 | NDCG@10 |
|-------|-----|-----------|---------|
| BM25 (baseline) | 0.3855 | 0.3747 | 0.3072 |
| BM25 + BPE | 0.4297 | 0.4501 | 0.3569 |
| Dense E5 | 0.7936 | 0.8642 | 0.7481 |
| Hybrid (BM25 + Dense) | 0.7936 | 0.8642 | 0.7481 | 
| finetunedE5 | 0.8061 | 0.8704 | 0.7631 |
| **Dense + Reranker** | **0.9019** | **0.9315** | **0.8642** |

## Архитектура
| Этап | Действие | Результат |
|------|----------|-----------|
| 1. | **Поисковый запрос** | |
| ↓ | | |
| 2. | **Dense Retriever (E5)** | → топ-k кандидатов |
| ↓ | | |
| 3. | **Cross-Encoder Reranker** | → переранжирование топ-m(m ≤ k) |
| ↓ | | |
| 4. | **Финальный результат** | |

Dense Retriever — intfloat/multilingual-e5-base. Кодирует тексты в векторы, ищет ближайших соседей через FAISS.

Reranker — BAAI/bge-reranker-v2-m3. Cross-encoder, который точнее оценивает пару (запрос, документ).

## Структура проекта
├── src/  
│   ├── data/           # загрузка MIRACL  
│   ├── models/         # BM25, Dense, Hybrid, Reranker  
│   ├── training/       # fine-tuning  
│   ├── evaluation/     # метрики  
│   └── inference.py    # основной класс для поиска  
├── configs/            # конфигурация  
├── scripts/            # скрипты  
├── notebooks/          # эксперименты  
└── requirements.txt  

## Эксперименты

Весь путь от baseline до лучшей модели задокументирован в ноутбуках:

1. 01_baseline.ipynb — загрузка данных, EDA, простой BM25  
2. 02_bm25_bpe.ipynb — BM25 с BPE токенизацией  
3. 03_dense_retriever.ipynb — Dense Retriever на E5  
4.  04_hybrid_reranker.ipynb — гибридный поиск + reranking  
5. 05_finetuning.ipynb — дообучение E5 на MIRACL  

## Дообучение

```python
import sys

from src import *
from configs.config import *

# Скачать индекс
download_dense_index()

# Загрузить данные
data = load_all_data()

# Загрузить retriever для майнинга негативов
dense = DenseRetrieverE5(E5_MODEL_NAME)
dense.load_index(DENSE_INDEX_PATH, data['doc_ids'], data['passages'])

# Подготовить данные
training_data = prepare_training_data(
    data['train_queries'],
    data['train_qrels'],
    data['corpus'],
    dense
)
dataset = E5Dataset(training_data)

# Обучить
train_model(E5_MODEL_NAME, dataset, "finetuned_e5")

# Оценить
trained = DenseRetrieverE5("finetuned_e5")
trained.fit(data['doc_ids'], data['passages'])

metrics = evaluate(trained, data['dev_queries'], data['dev_qrels'])
print(metrics)
```
## Инференс

```python 
from src import *

# Подготовка
download_dense_index()
data = load_all_data(target_docs=500_000)

# Создание (использует DenseRetrieverE5 + Reranker внутри)
retriever = FinalRetriever()
retriever.load_index("dense_e5.faiss", data['doc_ids'], data['passages'])

# Использование
print(retriever.answer("Кто такой Юрий Гагарин?"))

# С деталями
results = retriever.search("Столица России", top_k=3)
for r in results:
    print(f"[{r['score']:.4f}] {r['passage'][:100]}...")

# Без reranker (только Dense)
results = retriever.search("Столица России", top_k=3, use_reranker=False)
```

## Данные 
Используется датасет MIRACL(русская часть)

- везде использовалось 500K документов из корпуса  
- 4682 обучающих запросов
- 1251 тестовых запросов

## Скачивание
 Для скачивания предобученных файлов использовать функции из scripts\download_assets.py

 ## Технические детали

 ## Модели

| Компонент | Модель | Размер |
|-----------|--------|--------|
| Dense Retriever | `intfloat/multilingual-e5-base` | 278M параметров |
| Reranker | `BAAI/bge-reranker-v2-m3` | 568M параметров |
| BPE Tokenizer | SentencePiece | 25K токенов |

