from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, SlidingWindowRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ARCDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

ARC_c_reader_cfg = dict(
    input_columns=['question', 'textA', 'textB', 'textC', 'textD'],
    output_column='answerKey')

ARC_c_infer_cfg_0shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=
                    '問題：{question}\nA. {textA}\nB. {textB}\nC. {textC}\nD. {textD}\n由提供嘅選項中直接用選項嘅字母作答。\n 回應：'
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

ARC_c_infer_cfg_5shot = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='樣例：\n問題：{question}\nA. {textA}\nB. {textB}\nC. {textC}\nD. {textD}\n回應：{answerKey}\n'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=f'</E>\n問題：{{question}}\nA. {{textA}}\nB. {{textB}}\nC. {{textC}}\nD. {{textD}}\n由提供嘅選項中直接用選項嘅字母作答。\n回應：',
        ice_token='</E>',
    ),
    retriever=dict(type=SlidingWindowRetriever, k=5),
    inferencer=dict(type=GenInferencer),
)


ARC_c_eval_cfg_0shot = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)
ARC_c_eval_cfg_5shot = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

ARC_c_cantonese_datasets = [
    dict(
        abbr='ARC-c_0shot',
        type=ARCDataset,
        path='/cpfs01/user/chenpengan/root-2/opencompass/cantonese_dataset/translated_ARC_challenge.jsonl',
        name='challenge',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg_0shot,
        eval_cfg=ARC_c_eval_cfg_0shot,
    ),
        dict(
        abbr='ARC-c_5shot',
        type=ARCDataset,
        path='/cpfs01/user/chenpengan/root-2/opencompass/cantonese_dataset/translated_ARC_challenge.jsonl',
        name='challenge',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg_5shot,
        eval_cfg=ARC_c_eval_cfg_5shot,
    ),
]
