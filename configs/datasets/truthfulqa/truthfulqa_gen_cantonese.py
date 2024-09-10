from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever,BaseRetriever, SlidingWindowRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TruthfulQACantoneseDataset, TruthfulQACantoneseEvaluator

truthfulqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='reference',
    train_split='validation',
    test_split='validation'
)

truthfulqa_infer_cfg_5shot = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='樣例：\n問題：{question}\n回應：{best_answer}'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=f'</E>\n用粵語答下面問題：\n問題：{{question}}\n回應：\n',
        ice_token='</E>',
    ),
    retriever=dict(type=SlidingWindowRetriever, k=5),
    inferencer=dict(type=GenInferencer),
)
# truthfulqa_infer_cfg_5shot = dict(
#     ice_template=dict(
#         type=PromptTemplate,
#         template=dict(
#             round=[
#                 dict(role='HUMAN', prompt='樣例：\n問題：{question}\n'),
#                 dict(role='BOT', prompt='回應：{best_answer}\n')
#             ]
#         )
#     ),
#     prompt_template=dict(
#         type=PromptTemplate,
#         template=dict(
#             begin=[
#                 '</E>',
#             ],
#             round=[

#             ],
#             end=[
#                 dict(role='SYSTEM', fallback_role='HUMAN', prompt='\n\n用粵語答下面問題：\n'),
#                 dict(role='HUMAN', prompt='問題：{question}\n'),
#                 dict(role='BOT', prompt='回應：{best_answer}\n')
#             ],
#         ),
#         ice_token='</E>',
#     ),
#     retriever=dict(type=SlidingWindowRetriever, k=5),
#     inferencer=dict(type=GenInferencer),
# )

truthfulqa_infer_cfg_0shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='用粵語答下面問題：\n問題：{question}\n'),
                dict(role='BOT', prompt='回應：\n')
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)


truthfulqa_eval_cfg_5shot = dict(
    evaluator=dict(
        type=TruthfulQACantoneseEvaluator, metrics=('bleurt', 'rouge', 'bleu'), key='ENV'
    )
)

truthfulqa_eval_cfg_0shot = dict(
    evaluator=dict(
        type=TruthfulQACantoneseEvaluator, metrics=('bleurt', 'rouge', 'bleu'), key='ENV'
    ),
    pred_role='BOT',
)

truthfulqa_cantonese_datasets = [
    dict(
        abbr='truthful_qa_cantonese_0shot',
        type=TruthfulQACantoneseDataset,
        path='/mnt/petrelfs/chenpengan/opencompass/cantonese_dataset/TruthfulQA_cantonese.json',
        name='generation',
        reader_cfg=truthfulqa_reader_cfg,
        infer_cfg=truthfulqa_infer_cfg_0shot,
        eval_cfg=truthfulqa_eval_cfg_0shot
    ),
        dict(
        abbr='truthful_qa_cantonese_5shot',
        type=TruthfulQACantoneseDataset,
        path='/mnt/petrelfs/chenpengan/opencompass/cantonese_dataset/TruthfulQA_cantonese.json',
        name='generation',
        reader_cfg=truthfulqa_reader_cfg,
        infer_cfg=truthfulqa_infer_cfg_5shot,
        eval_cfg=truthfulqa_eval_cfg_5shot
    ),
]

