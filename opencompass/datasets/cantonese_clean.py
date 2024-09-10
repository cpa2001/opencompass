import json
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset

@LOAD_DATASET.register_module()
class CantoneseCleanDataDataset(BaseDataset):
    @staticmethod
    def load(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        if not isinstance(data, list) or not data:
            raise ValueError(f"Unexpected format in {path}. Expected a non-empty list of dictionaries.")
        
        if 'conversation' not in data[0]:
            raise ValueError(f"Data is missing required key 'conversation' in {path}.")
        
        human_texts = []
        assistant_texts = []
        for item in data:
            conversation = item.get('conversation', [])
            if conversation and isinstance(conversation, list):
                human_texts.append(conversation[0].get('human', ''))
                assistant_texts.append(conversation[0].get('assistant', ''))
            else:
                raise ValueError(f"Invalid conversation format in {path}.")
        
        processed_data = {
            'input': human_texts,
            'target': assistant_texts
        }
        
        dataset = Dataset.from_dict(processed_data)
        return DatasetDict({'train': dataset, 'test': dataset})

    @classmethod
    def get_dataset(cls, path):
        return cls.load(path)
