from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, SlidingWindowRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NLGEvaluator
from opencompass.datasets import ARCDatasetNLG

ARC_c_reader_cfg = dict(
    input_columns=['question', 'correct_answer', 'incorrect_answers'],
    output_column='answerKey',
)

ARC_c_infer_cfg_0shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Question: {question}\nAnswer: '
                )
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

ARC_c_infer_cfg_5shot = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='Example: \nQuestion: {question}\nAnswer: {correct_answer}\n'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=f'</E>\nQuestion: {{question}}\nAnswer: ',
        ice_token='</E>',
    ),
    retriever=dict(type=SlidingWindowRetriever, k=5),
    inferencer=dict(type=GenInferencer),
)

ARC_c_eval_cfg = dict(
    evaluator=dict(type=NLGEvaluator),
    pred_role='BOT',
)


ARC_c_NLG_english_datasets = [
    dict(
        abbr='ARC-c_0shot',
        type=ARCDatasetNLG,
        path='./data/ARC/ARC-c/ARC-Challenge-Test.jsonl',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg_0shot,
        eval_cfg=ARC_c_eval_cfg,
    ),
    dict(
        abbr='ARC-c_5shot',
        type=ARCDatasetNLG,
        path='./data/ARC/ARC-c/ARC-Challenge-Test.jsonl',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg_5shot,
        eval_cfg=ARC_c_eval_cfg,
    ),
]