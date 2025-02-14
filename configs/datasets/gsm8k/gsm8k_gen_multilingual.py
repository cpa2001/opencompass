# gsm8k_gen_multilingual.py
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, SlidingWindowRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MultilingualGSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

# Reader configuration
gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

# Language mappings and prompts
LANGUAGE_MAPPING = {
    'gu': 'Gujarati',
    'cs': 'Czech',
    # 'yue': 'Cantonese',
    # 'en': 'English',
}

LANGUAGE_PROMPTS = {
    'cs': "Přemýšlejte krok za krokem a označte konečnou odpověď pomocí '####'. Odpovězte na následující otázku v češtině:\nOtázka: {question}\nOdpověď:\n",
    'gu': "ક્રમશઃ વિચારો અને અંતિમ જવાબને '####' સાથે ચિહ્નિત કરો. નીચેના પ્રશ્નનો જવાબ ગુજરાતીમાં આપો:\nપ્રશ્ન: {question}\nજવાબ:\n",
    'en': "Please think step by step, mark the final answer with '####'. Answer the following question in English:\nQuestion: {question}\nAnswer:\n"
}

languages = list(LANGUAGE_MAPPING.keys())

gsm8k_multilingual_datasets = []

for lang in languages:
    lang_prompt = LANGUAGE_PROMPTS.get(lang, LANGUAGE_PROMPTS['en'])
    
    # 0-shot 推理配置
    gsm8k_infer_cfg_0shot = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt=lang_prompt),
                ],
            )),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=512))
    
    # 5-shot 推理配置
    gsm8k_infer_cfg_5shot = dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nQuestion: {question}\nResponse: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template=f"</E>\n{lang_prompt}",
            ice_token='</E>',
        ),
        retriever=dict(type=SlidingWindowRetriever, k=5),
        inferencer=dict(type=GenInferencer),
    )
    
    # 评估配置
    gsm8k_eval_cfg_0shot = dict(
        evaluator=dict(type=Gsm8kEvaluator),
        pred_postprocessor=dict(type=gsm8k_postprocess),
        dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
        pred_role='BOT',
    )
    
    gsm8k_eval_cfg_5shot = dict(
        evaluator=dict(type=Gsm8kEvaluator),
        pred_postprocessor=dict(type=gsm8k_postprocess),
        dataset_postprocessor=dict(type=gsm8k_dataset_postprocess)
    )
    
    # 添加 0-shot 和 5-shot 数据集配置
    gsm8k_multilingual_datasets.extend([
        dict(
            abbr=f'gsm8k_{lang}_0shot',
            type=MultilingualGSM8KDataset,
            path='/cpfs01/user/chenpengan/root-2/opencompass/data/gsm8k_multilingual',
            language=lang,
            reader_cfg=gsm8k_reader_cfg,
            infer_cfg=gsm8k_infer_cfg_0shot,
            eval_cfg=gsm8k_eval_cfg_0shot,
        ),
        dict(
            abbr=f'gsm8k_{lang}_5shot',
            type=MultilingualGSM8KDataset,
            path='/cpfs01/user/chenpengan/root-2/opencompass/data/gsm8k_multilingual',
            language=lang,
            reader_cfg=gsm8k_reader_cfg,
            infer_cfg=gsm8k_infer_cfg_5shot,
            eval_cfg=gsm8k_eval_cfg_5shot,
        )
    ])
