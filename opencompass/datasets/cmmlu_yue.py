import json
import os
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset

@LOAD_DATASET.register_module()
class CMMLUYueDataset(BaseDataset):
    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        raw_data = []
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path does not exist: {path}")
        
        json_files = [f for f in os.listdir(path) if f.endswith('.json')]
        if not json_files:
            raise ValueError(f"No JSON files found in the specified path: {path}")
        
        for filename in json_files:
            try:
                with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        print(f"Warning: Unexpected format in {filename}. Expected a list of dictionaries.")
                        continue
                    raw_data.extend(data)
            except json.JSONDecodeError:
                print(f"Error: Unable to parse JSON in file {filename}. Skipping this file.")
            except Exception as e:
                print(f"Unexpected error when processing {filename}: {str(e)}")
        
        if not raw_data:
            raise ValueError("No valid data found in any of the JSON files.")
        
        dataset['test'] = Dataset.from_list(raw_data)
        dataset['train'] = dataset['test']
        return dataset