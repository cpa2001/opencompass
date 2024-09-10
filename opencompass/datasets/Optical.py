import csv
import json
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset
from collections import Counter

@LOAD_DATASET.register_module()
class OpticalDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        raw_data = []
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                item = {
                    'id': str(i),
                    'question': row['question'],
                    'A': row['A'],
                    'B': row['B'],
                    'C': row['C'],
                    'D': row['D'],
                    'answer': row['answer'],
                }
                if 'explanation' in row:
                    item['explanation'] = row['explanation']
                raw_data.append(item)
        
        dataset['train'] = Dataset.from_list(raw_data)
        dataset['test'] = Dataset.from_list(raw_data)
        return dataset
