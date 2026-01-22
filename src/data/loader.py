"""Загрузка данных MIRACL"""

import os
import json
import gzip
import random
import requests
from collections import defaultdict
from tqdm.notebook import tqdm
from huggingface_hub import hf_hub_download, list_repo_files

from configs.config import BASE_URL, CORPUS_REPO, SEED


def load_queries(split: str) -> dict:
    """Загрузка запросов"""
    url = f"{BASE_URL}topics/topics.miracl-v1.0-ru-{split}.tsv"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        queries = {}
        for line in response.text.strip().split('\n')[1:]:
            parts = line.split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
        return queries
    except:
        return {}


def load_qrels(split: str) -> dict:
    """Загрузка qrels"""
    url = f"{BASE_URL}qrels/qrels.miracl-v1.0-ru-{split}.tsv"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        qrels = defaultdict(lambda: {'positive': [], 'negative': []})
        for line in response.text.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) >= 4:
                qid, doc_id, rel = parts[0], parts[2], int(parts[3])
                key = 'positive' if rel > 0 else 'negative'
                qrels[qid][key].append(doc_id)
        return dict(qrels)
    except:
        return {}


def get_required_doc_ids(qrels_dict: dict) -> set:
    """Собрать все doc_id из qrels"""
    doc_ids = set()
    for q in qrels_dict.values():
        doc_ids.update(q.get('positive', []))
        doc_ids.update(q.get('negative', []))
    return doc_ids


def load_corpus(required_doc_ids: set, target_docs: int = 500_000, seed: int = SEED) -> tuple:
    """Загрузка корпуса с гарантированным покрытием qrels"""
    random.seed(seed)
    
    all_files = list_repo_files(CORPUS_REPO, repo_type="dataset")
    jsonl_files = sorted([f for f in all_files if 'ru' in f and f.endswith('.jsonl.gz')])
    docs_per_file = target_docs // len(jsonl_files)

    corpus = {}
    doc_ids = []
    passages = []

    for jsonl_file in tqdm(jsonl_files, desc="Загрузка корпуса"):
        try:
            file_path = hf_hub_download(
                repo_id=CORPUS_REPO,
                filename=jsonl_file,
                repo_type="dataset"
            )

            file_docs = []
            required_docs = []

            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    doc = {
                        'docid': item['docid'],
                        'title': item['title'],
                        'text': item['text'],
                        'full_text': f"{item['title']}. {item['text']}"
                    }
                    
                    if item['docid'] in required_doc_ids:
                        required_docs.append(doc)
                    else:
                        file_docs.append(doc)

            sample_size = min(docs_per_file, len(file_docs))
            selected = random.sample(file_docs, sample_size) if file_docs else []
            
            for doc in selected + required_docs:
                if doc['docid'] not in corpus:
                    corpus[doc['docid']] = {
                        'title': doc['title'],
                        'text': doc['text'],
                        'full_text': doc['full_text']
                    }
                    doc_ids.append(doc['docid'])
                    passages.append(doc['full_text'])

        except Exception as e:
            print(f"Ошибка: {e}")

    return corpus, doc_ids, passages


def load_all_data(target_docs: int = 500_000):
    """Загрузка всех данных"""
    # Запросы и qrels
    dev_queries = load_queries('dev')
    dev_qrels = load_qrels('dev')
    train_queries = load_queries('train')
    train_qrels = load_qrels('train')
    
    all_qrels = {**dev_qrels, **train_qrels}
    required_doc_ids = get_required_doc_ids(all_qrels)
    
    # Корпус
    corpus, doc_ids, passages = load_corpus(required_doc_ids, target_docs)
    
    return {
        'dev_queries': dev_queries,
        'dev_qrels': dev_qrels,
        'train_queries': train_queries,
        'train_qrels': train_qrels,
        'corpus': corpus,
        'doc_ids': doc_ids,
        'passages': passages
    }