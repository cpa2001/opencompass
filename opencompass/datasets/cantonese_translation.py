import json
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset

@LOAD_DATASET.register_module()
class CantoneseTranslationDataset(BaseDataset):
    @staticmethod
    def load(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or not data:
            raise ValueError(f"Unexpected format in {path}. Expected a non-empty list of dictionaries.")
        
        input_key = 'en' if 'en' in data[0] else 'zh'
        if input_key not in data[0] or 'yue' not in data[0]:
            raise ValueError(f"Data is missing required keys. Expected '{input_key}' and 'yue'")
        
        processed_data = {
            'input': [item[input_key] for item in data],
            'target': [item['yue'] for item in data]
        }
        
        dataset = Dataset.from_dict(processed_data)
        return DatasetDict({'train': dataset, 'test': dataset})

    @classmethod
    def get_dataset(cls, path):
        return cls.load(path)