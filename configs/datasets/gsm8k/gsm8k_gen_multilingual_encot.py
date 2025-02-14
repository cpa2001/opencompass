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

# LANGUAGE_PROMPTS = {
#     'cs': "Nejprve to vyřešme krok za krokem:\n{question}\n\nThinking process in English:\n1) [Your step-by-step reasoning will go here]\n2) ...\n3) ...\n\nA nyní poskytnu konečnou odpověď v češtině:\nOdpověď:\n####",
    
#     'gu': "ચાલો સૌ પ્રથમ આ સમસ્યાને સ્ટેપ બાય સ્ટેપ હલ કરીએ:\n{question}\n\nThinking process in English:\n1) [Your step-by-step reasoning will go here]\n2) ...\n3) ...\n\nહવે હું ગુજરાતીમાં અંતિમ જવાબ આપીશ:\nજવાબ:\n####",
    
#     'en': "Let's solve this step by step:\n{question}\n\nThinking process:\n1) [Your step-by-step reasoning will go here]\n2) ...\n3) ...\n\nFinal answer:\n####"
# }
# LANGUAGE_PROMPTS = {
#     'cs': "Let's first solve this problem step by step in English. Focus on thinking clearly and logically in English before deriving the final answer in Czech.\n\nProblem:\n{question}\n\nStep-by-step reasoning (in English):\n1) [Start your detailed reasoning in English here]\n2) ...\n3) ...\n\nNow provide the final answer in Czech:\nOdpověď:\n####",
    
#     'gu': "Let's first solve this problem step by step in English. Focus on thinking clearly and logically in English before deriving the final answer in Gujarati.\n\nProblem:\n{question}\n\nStep-by-step reasoning (in English):\n1) [Start your detailed reasoning in English here]\n2) ...\n3) ...\n\nNow provide the final answer in Gujarati:\nજવાબ:\n####",
    
#     'en': "Let's solve this step by step:\n{question}\n\nStep-by-step reasoning (in English):\n1) [Your detailed reasoning will go here]\n2) ...\n3) ...\n\nFinal answer:\n####"
# }
# LANGUAGE_PROMPTS = {
#     'cs': "Let's translate this Czech question to English first, then solve it in English.\n\nOriginal question:\n{question}\n\nEnglish translation:\n[translate the question]\n\nStep-by-step solution:\n1) [reason in English]\n2) ...\n3) ...\n\nFinal answer:\n#### [final numerical answer]",
    
#     'gu': "Let's translate this Gujarati question to English first, then solve it in English.\n\nOriginal question:\n{question}\n\nEnglish translation:\n[translate the question]\n\nStep-by-step solution:\n1) [reason in English]\n2) ...\n3) ...\n\nFinal answer:\n#### [final numerical answer]",

#     'en': "Question:\n{question}\n\nStep-by-step solution:\n1) [reason in English]\n2) ...\n3) ...\n\nFinal answer:\n#### [final numerical answer]"
# }
LANGUAGE_PROMPTS = {
    'cs': """Let's solve this GSM8K math problem in three steps:

Step 1: Translate to English
Original Czech question: {question}
English translation: [translate here]

Step 2: Solve step by step in English
1) [first logical step]
2) [second logical step]
3) [continue steps as needed...]
4) Therefore, the answer is [explain final calculation]

Step 3: Translate solution to Czech
[Translate the above solution steps to Czech]

Final answer:
#### [numerical answer]""",
    
    'gu': """Let's solve this GSM8K math problem in three steps:

Step 1: Translate to English
Original Gujarati question: {question}
English translation: [translate here]

Step 2: Solve step by step in English
1) [first logical step]
2) [second logical step]
3) [continue steps as needed...]
4) Therefore, the answer is [explain final calculation]

Step 3: Translate solution to Gujarati
[Translate the above solution steps to Gujarati]

Final answer:
#### [numerical answer]""",

    'en': """Let's solve this GSM8K math problem:

Question: {question}

Step-by-step solution in English:
1) [first logical step]
2) [second logical step]
3) [continue steps as needed...]
4) Therefore, the answer is [explain final calculation]

Final answer:
#### [numerical answer]"""
}

languages = list(LANGUAGE_MAPPING.keys())

gsm8k_multilingual_encot_datasets = []

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
    gsm8k_multilingual_encot_datasets.extend([
        dict(
            abbr=f'gsm8k_{lang}_0shot',
            type=MultilingualGSM8KDataset,
            path='/cpfs02/shared/optimal/chenpengan/opencompass/data/gsm8k_multilingual',
            language=lang,
            reader_cfg=gsm8k_reader_cfg,
            infer_cfg=gsm8k_infer_cfg_0shot,
            eval_cfg=gsm8k_eval_cfg_0shot,
        ),
        dict(
            abbr=f'gsm8k_{lang}_5shot',
            type=MultilingualGSM8KDataset,
            path='/cpfs02/shared/optimal/chenpengan/opencompass/data/gsm8k_multilingual',
            language=lang,
            reader_cfg=gsm8k_reader_cfg,
            infer_cfg=gsm8k_infer_cfg_5shot,
            eval_cfg=gsm8k_eval_cfg_5shot,
        )
    ])
