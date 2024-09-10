from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, SlidingWindowRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NLGEvaluator
from opencompass.datasets import ARCDatasetNLG

ARC_c_reader_cfg = dict(
    input_columns=['question', 'correct_answer', 'incorrect_answers', 'answerKey'],
    output_column='answerKey',
)

ARC_c_infer_cfg_0shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='問題：{question}\n回應：'
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
        template='問題：{question}\n回應：{correct_answer}\n'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=f'</E>\n問題：{{question}}\n回應：',
        ice_token='</E>',
    ),
    retriever=dict(type=SlidingWindowRetriever, k=5),
    inferencer=dict(type=GenInferencer),
)

# ARC_c_infer_cfg_0shot = dict(
#     prompt_template=dict(
#         type=PromptTemplate,
#         template=dict(
#             round=[
#                 dict(
#                     role='HUMAN',
#                     prompt='問題：{question}\n正確答案：{correct_answer}\n錯誤答案：{incorrect_answers}\n回應：'
#                 )
#             ],
#         ),
#     ),
#     retriever=dict(type=ZeroRetriever),
#     inferencer=dict(type=GenInferencer),
# )

# ARC_c_infer_cfg_5shot = dict(
#     ice_template=dict(
#         type=PromptTemplate,
#         template='問題：{question}\n正確答案：{correct_answer}\n錯誤答案：{incorrect_answers}\n回應：{correct_answer}\n'
#     ),
#     prompt_template=dict(
#         type=PromptTemplate,
#         template=f'</E>\n問題：{{question}}\n回應：',
#         ice_token='</E>',
#     ),
#     retriever=dict(type=SlidingWindowRetriever, k=5),
#     inferencer=dict(type=GenInferencer),
# )

ARC_c_eval_cfg = dict(
    evaluator=dict(type=NLGEvaluator),
    pred_role='BOT',
    dataset_role='HUMAN',
)

ARC_c_NLG_cantonese_datasets = [
    dict(
        abbr='ARC-c_0shot',
        type=ARCDatasetNLG,
        path='/mnt/petrelfs/chenpengan/opencompass/cantonese_dataset/translated_ARC_challenge.jsonl',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg_0shot,
        eval_cfg=ARC_c_eval_cfg,
    ),
    dict(
        abbr='ARC-c_5shot',
        type=ARCDatasetNLG,
        path='/mnt/petrelfs/chenpengan/opencompass/cantonese_dataset/translated_ARC_challenge.jsonl',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg_5shot,
        eval_cfg=ARC_c_eval_cfg,
    ),
]