from abc import abstractstaticmethod
from typing import Dict, Optional, Union

from datasets import Dataset, DatasetDict

from opencompass.openicl import DatasetReader


class BaseDataset:

    def __init__(self, reader_cfg: Optional[Dict] = {}, **kwargs):
        self.dataset = self.load(**kwargs)
        self._init_reader(**reader_cfg)

    def _init_reader(self, **kwargs):
        self.reader = DatasetReader(self.dataset, **kwargs)

    @property
    def train(self):
        return self.reader.dataset['train']
        # return self.reader.dataset

    @property
    def test(self):
        return self.reader.dataset['test']
        # return self.reader.dataset

    @abstractstaticmethod
    def load(**kwargs) -> Union[Dataset, DatasetDict]:
        pass
