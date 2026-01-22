"""Скрипт для запуска дообучения"""

import os
import sys
import argparse
import gc
import torch

# Добавляем корень проекта
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.data.loader import load_all_data
from src.data.dataset import E5Dataset
from src.models.dense import DenseRetrieverE5
from src.training.trainer import prepare_training_data, train_model
from src.evaluation.metrics import evaluate, print_comparison
from configs.config import (
    E5_MODEL_NAME, 
    DENSE_INDEX_PATH,
    TARGET_DOCS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)
from scripts.download_assets import download_all


def main(args):
    # 1. Скачивание файлов
    print("="*60)
    print("ДООБУЧЕНИЕ E5 НА MIRACL")
    print("="*60)
    
    download_all()
    
    # 2. Загрузка данных
    print("\n1. Загрузка данных...")
    data = load_all_data(target_docs=args.target_docs)
    print(f"   Документов: {len(data['corpus']):,}")
    print(f"   Train запросов: {len(data['train_queries'])}")
    
    # 3. Dense Retriever для майнинга
    print("\n2. Загрузка Dense Retriever...")
    dense_retriever = DenseRetrieverE5(E5_MODEL_NAME)
    dense_retriever.load_index(
        DENSE_INDEX_PATH,
        data['doc_ids'],
        data['passages']
    )
    
    # 4. Подготовка данных
    print("\n3. Подготовка обучающих данных...")
    training_data = prepare_training_data(
        queries=data['train_queries'],
        qrels=data['train_qrels'],
        corpus=data['corpus'],
        retriever=dense_retriever
    )
    train_dataset = E5Dataset(training_data, max_negatives=3)
    print(f"   Примеров: {len(train_dataset):,}")
    
    # 5. Обучение
    print("\n4. Обучение...")
    trained_model = train_model(
        model_name=E5_MODEL_NAME,
        train_dataset=train_dataset,
        output_path=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # 6. Очистка памяти
    del dense_retriever, trained_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 7. Оценка
    print("\n5. Оценка...")
    trained_retriever = DenseRetrieverE5(model_name=args.output)
    trained_retriever.fit(data['doc_ids'], data['passages'], batch_size=64)
    trained_retriever.save_index(f"{args.output}.faiss")
    
    trained_metrics = evaluate(trained_retriever, data['dev_queries'], data['dev_qrels'])
    
    baseline = {'MRR': 0.7935, 'NDCG@10': 0.7479, 'Recall@10': 0.8641}
    print_comparison(baseline, trained_metrics, "Fine-tuned")
    
    print("\n" + "="*60)
    print("ГОТОВО!")
    print(f"Модель сохранена: {args.output}/")
    print(f"Индекс сохранён: {args.output}.faiss")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Дообучение E5 на MIRACL")
    
    parser.add_argument("--output", "-o", default="finetuned_e5", 
                        help="Путь для сохранения модели")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE,
                        help="Размер батча")
    parser.add_argument("--epochs", "-e", type=int, default=EPOCHS,
                        help="Количество эпох")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--target-docs", type=int, default=TARGET_DOCS,
                        help="Количество документов в корпусе")
    
    args = parser.parse_args()
    main(args)