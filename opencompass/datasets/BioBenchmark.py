import json
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset

@LOAD_DATASET.register_module()
class BioBenchmarkDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Handle both list and dict formats
        if isinstance(raw_data, list):
            data_list = raw_data
        elif isinstance(raw_data, dict):
            data_list = []
            for key in raw_data:
                if isinstance(raw_data[key], list):
                    data_list.extend(raw_data[key])
                else:
                    raise ValueError(f"Expected a list under key '{key}', but got {type(raw_data[key])}")
        else:
            raise ValueError(f"Unsupported data format: {type(raw_data)}")

        # Add free_form_answer if available
        for item in data_list:
            if 'answer' in item and isinstance(item['answer'], dict):
                item['free_form_answer'] = item['answer'].get('free_form_answer', None)

        dataset['train'] = Dataset.from_list(data_list)
        dataset['test'] = Dataset.from_list(data_list)
        return dataset
