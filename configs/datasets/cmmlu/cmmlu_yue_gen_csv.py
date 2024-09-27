from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, SlidingWindowRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess
from opencompass.datasets import CMMLUDataset  

cmmlu_yue_subject_mapping = {
    'arts': '藝術學',
    'chinese_civil_service_exam': '中國公務員考試',
    'chinese_literature': '中國文學',
    'college_medicine': '大學醫學',
    'economics': '經濟學',
    'education': '教育學',
    'electrical_engineering': '電氣工程',
    'ethnology': '民族學',
    'high_school_geography': '高中地理',
    'journalism': '新聞學',
    'logical': '邏輯學',
    'machine_learning': '機器學習',
    'management': '管理學',
    'marketing': '市場營銷',
    'marxist_theory': '馬克思主義理論',
    'philosophy': '哲學',
    'professional_psychology': '專業心理學',
    'security_study': '安全研究',
    'sociology': '社會學',
    'sports_science': '體育學',
    'world_history': '世界歷史',
    'world_religions': '世界宗教'
}

cmmlu_yue_all_sets = list(cmmlu_yue_subject_mapping.keys())

cmmlu_yue_gen_datasets = []
for _name in cmmlu_yue_all_sets:
    _ch_name = cmmlu_yue_subject_mapping[_name]
    
    # 5-shot configuration
    cmmlu_yue_infer_cfg_5shot = dict(
        ice_template=dict(
            type=PromptTemplate,
            template='樣例：\n問題：{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n回應：{answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template=f'</E>\n以下係關於{_ch_name}嘅單項選擇題，請直接畀出正確答案嘅選項。\n問題：{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案：',
            ice_token='</E>',
        ),
        retriever=dict(type=SlidingWindowRetriever, k=5),
        inferencer=dict(type=GenInferencer),
    )
    
    # 0-shot configuration
    cmmlu_yue_infer_cfg_0shot = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=f'以下係關於{_ch_name}嘅單項選擇題，請直接畀出正確答案嘅選項。\n問題：{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}'
                    ),
                    dict(role='BOT', prompt='答案係: {answer}\n'),
                ]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    cmmlu_yue_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=first_capital_postprocess))

    # 5-shot dataset
    cmmlu_yue_gen_datasets.append(
        dict(
            type=CMMLUDataset,
            path='/mnt/petrelfs/chenpengan/opencompass/cantonese_dataset/CMMLU_yue',
            name=_name,
            abbr=f'cmmlu-yue-{_name}-5shot',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='test',
                test_split='test'),
            infer_cfg=cmmlu_yue_infer_cfg_5shot,
            eval_cfg=cmmlu_yue_eval_cfg,
        ))
    
    # 0-shot dataset
    cmmlu_yue_gen_datasets.append(
        dict(
            type=CMMLUDataset,
            path='/mnt/petrelfs/chenpengan/opencompass/cantonese_dataset/CMMLU_yue',
            name=_name,
            abbr=f'cmmlu-yue-{_name}-0shot',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='test',
                test_split='test'),
            infer_cfg=cmmlu_yue_infer_cfg_0shot,
            eval_cfg=cmmlu_yue_eval_cfg,
        ))

del _name, _ch_name
