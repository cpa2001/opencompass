from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.cantonese_clean import CantoneseCleanDataDataset

reader_cfg = dict(input_columns='input', output_column='target')

infer_cfg_0shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='',
            round=[
                dict(role='HUMAN', prompt="請將下面呢句/段話直接翻譯成粵語：\n'''{input}'''"),
                dict(role='BOT', prompt=''),
            ]
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

clean_yue_dataset = {
    'abbr': 'clean_yue_dataset',
    'type': CantoneseCleanDataDataset,
    'path': '/mnt/petrelfs/chenpengan/opencompass/cantonese_dataset/decoded_math.jsonl',
    'reader_cfg': reader_cfg,
    'infer_cfg': infer_cfg_0shot
}

cfg = clean_yue_dataset

