# File: gsm8k_translation.py

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import SlidingWindowRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import gsm8k_postprocess, Gsm8kEvaluator, GSM8KTranslateDataset

gsm8k_translate_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='Example:\nTranslate the following question to {target_language}:\n{question}\nTranslation: {translation_ground_truth}\n'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template='</E>\nBased on the examples above, please translate the following question into {target_language}, maintaining the original meaning while using idiomatic {target_language} expressions:\n\nEnglish: {question}\n\n{target_language} translation:',
        ice_token='</E>',
    ),
    retriever=dict(type=SlidingWindowRetriever, k=5),
    inferencer=dict(type=GenInferencer, max_out_len=512))

gsm8k_translate_datasets = [
    dict(
        abbr='gsm8k_translate_cantonese_5shot',
        type='GSM8KTranslateDataset',
        path='/mnt/petrelfs/chenpengan/opencompass/data/gsm8k',
        reader_cfg=dict(input_columns=['question'], output_column='translation_ground_truth'),
        translate_cfg=dict(
            target_language='Cantonese',
            ground_truth_path='/mnt/petrelfs/chenpengan/opencompass/cantonese_dataset/gsm8k_cantonese/test.jsonl'
        ),
        infer_cfg=gsm8k_translate_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(type=Gsm8kEvaluator),
            pred_postprocessor=dict(type='gsm8k_translate_postprocess'),
        )),
    dict(
        abbr='gsm8k_translate_chinese_5shot',
        type='GSM8KTranslateDataset',
        path='/mnt/petrelfs/chenpengan/opencompass/data/gsm8k',
        reader_cfg=dict(input_columns=['question'], output_column='translation_ground_truth'),
        translate_cfg=dict(
            target_language='Chinese',
            ground_truth_path='/mnt/petrelfs/chenpengan/opencompass/data/gsm8k_zh/test.jsonl'
        ),
        infer_cfg=gsm8k_translate_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(type=Gsm8kEvaluator),
            pred_postprocessor=dict(type='gsm8k_translate_postprocess'),
        )),
]