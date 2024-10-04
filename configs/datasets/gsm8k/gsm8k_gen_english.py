from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever,SlidingWindowRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg_0shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="Please think step by step, mark the final answer with '####'. Answer the following question in English:\nQuestion: {question}\nAnswer the question in English:\n"),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

gsm8k_infer_cfg_5shot = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='Example:\nQuestion: {question}\nResponse: {answer}\n'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=f"</E>\nPlease think step by step, mark the final answer with '####'. Answer the following question in English:\nQuestion: {{question}}\nAnswer the question in English:\n",
        ice_token='</E>',
    ),
    retriever=dict(type=SlidingWindowRetriever, k=5),
    inferencer=dict(type=GenInferencer),
)

gsm8k_eval_cfg_0shot = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
                      pred_role='BOT',)

gsm8k_eval_cfg_5shot = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_english_datasets = [
    dict(
        abbr='gsm8k_english_0shot',
        type=GSM8KDataset,
        path='/cpfs01/user/chenpengan/root-2/opencompass/data/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg_0shot,
        eval_cfg=gsm8k_eval_cfg_0shot),
    dict(
        abbr='gsm8k_english_5shot',
        type=GSM8KDataset,
        path='/cpfs01/user/chenpengan/root-2/opencompass/data/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg_5shot,
        eval_cfg=gsm8k_eval_cfg_5shot),        
]
