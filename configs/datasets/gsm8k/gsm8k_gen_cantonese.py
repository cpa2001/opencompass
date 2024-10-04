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
                dict(role='HUMAN', prompt="請逐步思考，最終答案前用「####」標記。用粵語答下面問題：\n問題：{question}\n用粵語回答問題：\n"),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

gsm8k_infer_cfg_5shot = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='樣例：\n問題：{question}\n回應：{answer}\n'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=f'</E>\n請逐步思考，最終答案前用「####」標記。用粵語答下面問題：\n問題：{{question}}\n用粵語回答問題：\n',
        ice_token='</E>',
    ),
    retriever=dict(type=SlidingWindowRetriever, k=5),
    inferencer=dict(type=GenInferencer),
    # ice_template=dict(
    #     type=PromptTemplate,
    #     template=dict(
    #         round=[
    #             dict(role='HUMAN', prompt='樣例：\n問題：{question}\n'),
    #             dict(role='BOT', prompt='回應：{answer}\n')
    #         ]
    #     )
    # ),
    # prompt_template=dict(
    #     type=PromptTemplate,
    #     template=dict(
    #         begin=[
    #             '</E>',
    #         ],
    #         round=[
    #             dict(role='SYSTEM', fallback_role='HUMAN', prompt='\n\n用粵語答下面問題：\n'),
    #             dict(role='HUMAN', prompt='問題：{question}\n'),
    #             dict(role='BOT', prompt='我地一步一步諗\n回應：{answer}\n')
    #         ],
    #         end=[
    #             # dict(role='SYSTEM', fallback_role='HUMAN', prompt='\n\n用粵語答下面問題：\n'),
    #             # dict(role='HUMAN', prompt='問題：{question}\n'),
    #             # dict(role='BOT', prompt='我地一步一步諗\n回應：{answer}\n')
    #         ],
    #     ),
    #     ice_token='</E>',
    # ),
    # retriever=dict(type=SlidingWindowRetriever,k=5),
    # inferencer=dict(type=GenInferencer, max_out_len=512)
)

gsm8k_eval_cfg_0shot = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
                      pred_role='BOT',)

gsm8k_eval_cfg_5shot = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_cantonese_datasets = [
    dict(
        abbr='gsm8k_cantonese_0shot',
        type=GSM8KDataset,
        path='/cpfs01/user/chenpengan/root-2/opencompass/cantonese_dataset/gsm8k_cantonese',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg_0shot,
        eval_cfg=gsm8k_eval_cfg_0shot),
    dict(
        abbr='gsm8k_cantonese_5shot',
        type=GSM8KDataset,
        path='/cpfs01/user/chenpengan/root-2/opencompass/cantonese_dataset/gsm8k_cantonese',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg_5shot,
        eval_cfg=gsm8k_eval_cfg_5shot),        
]
