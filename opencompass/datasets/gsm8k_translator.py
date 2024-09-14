import json
from opencompass.registry import LOAD_DATASET
from opencompass.datasets import GSM8KDataset
from opencompass.utils import get_data_path
from datasets import Dataset, DatasetDict

@LOAD_DATASET.register_module()
class GSM8KTranslateDataset(GSM8KDataset):
    def __init__(self, path, translate_cfg, reader_cfg=None, **kwargs):
        self.path = path
        self.translate_cfg = translate_cfg
        self.reader_cfg = reader_cfg or {}
        self.dataset = self.load(path=path, translate_cfg=translate_cfg, **kwargs)
        self.reader = self._create_reader()

    @classmethod
    def load(cls, path, translate_cfg=None, **kwargs):
        if path is None:
            raise ValueError("Path must be provided")

        path = get_data_path(path)
        dataset = super().load(path)
        
        if translate_cfg:
            with open(translate_cfg['ground_truth_path'], 'r', encoding='utf-8') as f:
                translations = [json.loads(line)['question'] for line in f]
            
            test_data = dataset['test'].to_dict()
            if len(translations) != len(test_data['question']):
                raise ValueError(f"Number of translations ({len(translations)}) does not match number of test questions ({len(test_data['question'])})")
            
            test_data['translation_ground_truth'] = translations
            test_data['target_language'] = [translate_cfg['target_language']] * len(test_data['question'])
            
            return DatasetDict({'test': Dataset.from_dict(test_data)})
        
        return DatasetDict({'test': dataset['test']})

    def _create_reader(self):
        input_columns = self.reader_cfg.get('input_columns', ['question'])
        output_column = self.reader_cfg.get('output_column', 'translation_ground_truth')
        
        class Reader:
            def __init__(self, dataset, input_columns, output_column):
                self.dataset = dataset
                self.input_columns = input_columns
                self.output_column = output_column

            def __getitem__(self, index):
                item = self.dataset['test'][index]
                inputs = {col: item[col] for col in self.input_columns if col in item}
                output = item[self.output_column] if self.output_column in item else ""
                return inputs, output

            def __len__(self):
                return len(self.dataset['test'])

        return Reader(self.dataset, input_columns, output_column)

    @property
    def train(self):
        return self.dataset['test']

    @property
    def test(self):
        return self.dataset['test']

def gsm8k_translate_postprocess(predictions, dataset):
    results = []
    for pred, orig_question, gt in zip(predictions, dataset['question'], dataset['translation_ground_truth']):
        results.append({
            'original_question': orig_question,
            'model_translation': pred,
            'ground_truth': gt
        })
    return results