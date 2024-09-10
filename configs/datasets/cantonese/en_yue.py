from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, BM25Retriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets.cantonese_translation import CantoneseTranslationDataset

# Common reader configuration
reader_cfg = dict(input_columns='input', output_column='target')

# 0-shot configuration
infer_cfg_0shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='',
            round=[
                dict(role='HUMAN', prompt='請將下面呢句/段話直接翻譯成粵語：\n{input}'),
                dict(role='BOT', prompt=''),
            ]
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

# 5-shot configuration
infer_cfg_5shot = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='樣例：\n請將下面呢句/段話直接翻譯成粵語：\n{input}\n翻譯：{target}\n'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=f'</E>\n根據上面嘅例子，請將下面呢句/段話直接翻譯成粵語：\n{{input}}\n',
        ice_token='</E>',
    ),
    # retriever=dict(type=SlidingWindowRetriever, k=5),
    retriever=dict(type=BM25Retriever, ice_num=5),
    inferencer=dict(type=GenInferencer),
)

en_yue_datasets = [
    dict(
        abbr='en-yue_0shot',
        type=CantoneseTranslationDataset,
        path='/mnt/petrelfs/chenpengan/opencompass/cantonese_dataset/en-yue.json',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg_0shot),
    dict(
        abbr='en-yue_5shot',
        type=CantoneseTranslationDataset,
        path='/mnt/petrelfs/chenpengan/opencompass/cantonese_dataset/en-yue.json',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg_5shot),
]
