from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever,SlidingWindowRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TruthfulQADataset, TruthfulQAEvaluator

truthfulqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='reference',
    train_split='validation',
    test_split='validation')

truthfulqa_infer_cfg_5shot = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='Example:\nQuestion: {question}\nResponse: {best_answer}'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=f'</E>\nAnswer the following question in English:\nQuestion: {{question}}\nResponse:\n',
        ice_token='</E>',
    ),
    retriever=dict(type=SlidingWindowRetriever, k=5),
    inferencer=dict(type=GenInferencer),
)
truthfulqa_infer_cfg_0shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Answer the following question in English:\nQuestion: {question}\n'),
                dict(role='BOT', prompt='Response:\n')
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

truthfulqa_eval_cfg_5shot = dict(
    evaluator=dict(
        type=TruthfulQAEvaluator, metrics=('bleurt', 'rouge', 'bleu'), key='ENV'
    )
)

truthfulqa_eval_cfg_0shot = dict(
    evaluator=dict(
        type=TruthfulQAEvaluator, metrics=('bleurt', 'rouge', 'bleu'), key='ENV'
    ),
    pred_role='BOT',
)

truthfulqa_english_datasets = [
    dict(
        abbr='truthful_qa_english_0shot',
        type=TruthfulQADataset,
        path='truthful_qa',
        name='generation',
        reader_cfg=truthfulqa_reader_cfg,
        infer_cfg=truthfulqa_infer_cfg_0shot,
        eval_cfg=truthfulqa_eval_cfg_0shot),
    dict(
        abbr='truthful_qa_english_5shot',
        type=TruthfulQADataset,
        path='truthful_qa',
        name='generation',
        reader_cfg=truthfulqa_reader_cfg,
        infer_cfg=truthfulqa_infer_cfg_5shot,
        eval_cfg=truthfulqa_eval_cfg_5shot)
]
