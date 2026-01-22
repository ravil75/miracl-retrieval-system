"""Функции для обучения"""

import gc
import time
import random
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer, losses
from tqdm.notebook import tqdm

from configs.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_SEQ_LENGTH, WARMUP_RATIO


def prepare_training_data(queries: dict, qrels: dict, corpus: dict, retriever) -> list:
    """Подготовка данных с hard negatives"""
    training_data = []
    corpus_keys = list(corpus.keys())
    
    valid_queries = {k: v for k, v in queries.items() if k in qrels}
    
    for qid, query_text in tqdm(valid_queries.items(), desc="Подготовка данных"):
        pos_ids = [pid for pid in qrels[qid]['positive'] if pid in corpus]
        if not pos_ids:
            continue
        
        # Hard negatives
        hard_negs = []
        try:
            hits = retriever.search(query_text, top_k=20)
            hard_negs = [h['doc_id'] for h in hits 
                        if h['doc_id'] not in pos_ids and h['doc_id'] in corpus]
        except:
            pass
        
        # Random negatives
        random_negs = []
        while len(random_negs) < 3:
            rid = random.choice(corpus_keys)
            if rid not in pos_ids and rid not in hard_negs:
                random_negs.append(rid)
        
        negatives = hard_negs[:5] + random_negs
        
        if negatives:
            for pid in pos_ids:
                training_data.append({
                    'query': query_text,
                    'positive': corpus[pid]['full_text'],
                    'negatives': [corpus[n]['full_text'] for n in negatives]
                })
    
    print(f"Примеров: {len(training_data):,}")
    return training_data


def train_model(
    model_name: str,
    train_dataset,
    output_path: str,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    max_seq_length: int = MAX_SEQ_LENGTH
):
    """Обучение модели"""
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Загрузка модели: {model_name}")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=0,
        collate_fn=model.smart_batching_collate
    )
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    scaler = GradScaler()
    
    print(f"\nКонфигурация:")
    print(f"  Device: {device}")
    print(f"  Batch: {batch_size}")
    print(f"  Steps: {total_steps}")
    
    print(f"\nСтарт обучения: {time.strftime('%H:%M:%S')}")
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, (batch_features, labels) in enumerate(pbar):
            features = [{k: v.to(device) for k, v in component.items()} 
                       for component in batch_features]
            
            optimizer.zero_grad()
            
            with autocast():
                loss_value = train_loss(features, labels)
            
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss_value.item()
            
            if step % 10 == 0:
                pbar.set_postfix({'loss': f"{loss_value.item():.4f}"})
            
            if step % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
    
    print(f"\nОбучение завершено за {(time.time()-start_time)/60:.1f} мин")
    
    model.save(output_path)
    print(f"Модель сохранена: {output_path}")
    
    return model