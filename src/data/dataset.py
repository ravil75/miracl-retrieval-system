"""Dataset классы для обучения"""

import random
from torch.utils.data import Dataset
from sentence_transformers import InputExample


class E5Dataset(Dataset):
    """Dataset для обучения E5 с hard negatives"""
    
    def __init__(self, data: list, max_negatives: int = 3):
        self.data = data
        self.max_negatives = max_negatives
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Префиксы для E5
        query = f"query: {item['query']}"
        positive = f"passage: {item['positive']}"
        
        # Случайные негативы
        negs = item['negatives'].copy()
        random.shuffle(negs)
        selected_negs = [f"passage: {n}" for n in negs[:self.max_negatives]]
        
        return InputExample(texts=[query, positive] + selected_negs)


class E5PairDataset(Dataset):
    """Простой Dataset: только (query, positive) пары"""
    
    def __init__(self, data: list):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        query = f"query: {item['query']}"
        positive = f"passage: {item['positive']}"
        
        return InputExample(texts=[query, positive])