from mmengine.config import read_base
from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .datasets.collections.chat_medium import datasets
    from .summarizers.medium import summarizer
    from .datasets.OpticalDataset.OpticalDataset_gen import optical_datasets
    from .datasets.truthfulqa.truthfulqa_gen_cantonese import truthfulqa_cantonese_datasets
    from .datasets.truthfulqa.truthfulqa_gen_english import truthfulqa_english_datasets
    from .datasets.gsm8k.gsm8k_gen_cantonese import gsm8k_cantonese_datasets
    from .datasets.gsm8k.gsm8k_gen_english import gsm8k_english_datasets
    from .datasets.cmmlu.cmmlu_gen_test import cmmlu_gen_datasets
    from .datasets.cmmlu.cmmlu_yue_gen_csv import cmmlu_yue_gen_datasets
    from .datasets.ARC_c.ARC_c_gen_cantonese import ARC_c_cantonese_datasets
    from .datasets.cantonese.en_yue import en_yue_datasets
    from .datasets.cantonese.zh_yue import zh_yue_datasets
    from .datasets.wmt19.wmt19_gen import wmt19_datasets
    

work_dir = 'outputs/wmt19/'
# datasets = [*ARC_c_cantonese_datasets]
# datasets = [*en_yue_datasets, *zh_yue_datasets]
# datasets = [*cmmlu_yue_gen_datasets]
datasets = [*wmt19_datasets]
# # GPT4 needs a special humaneval postprocessor
# for _dataset in datasets:
#     if _dataset['path'] == 'openai_humaneval':
#         _dataset['eval_cfg']['pred_postprocessor']['type'] = truthfulqa_datasets


# api_meta_template = dict(
#     round=[
#             dict(role='HUMAN', api_role='HUMAN'),
#             dict(role='BOT', api_role='BOT', generate=True),
#     ],
# )

# needs a special postprocessor for all except 'gsm8k' and 'strategyqa'
# from opencompass.utils import general_eval_wrapper_postprocess
# for _dataset in datasets:
#     if _dataset['abbr'] not in ['gsm8k', 'strategyqa']:
#         if hasattr(_dataset['eval_cfg'], 'pred_postprocessor'):
#             _dataset['eval_cfg']['pred_postprocessor']['postprocess'] = _dataset['eval_cfg']['pred_postprocessor']['type']
#             _dataset['eval_cfg']['pred_postprocessor']['type'] = general_eval_wrapper_postprocess
#         else:
#             # _dataset['eval_cfg']['pred_postprocessor'] = {'type': general_eval_wrapper_postprocess}
#             pass
from opencompass.utils import general_eval_wrapper_postprocess
for _dataset in datasets:
    if 'eval_cfg' in _dataset:
        if _dataset['abbr'] not in ['gsm8k', 'strategyqa']:
            if 'pred_postprocessor' in _dataset['eval_cfg']:
                _dataset['eval_cfg']['pred_postprocessor']['postprocess'] = _dataset['eval_cfg']['pred_postprocessor']['type']
                _dataset['eval_cfg']['pred_postprocessor']['type'] = general_eval_wrapper_postprocess


api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

GPT4 = [
    dict(abbr='gpt-4-0125',
        type=OpenAI, path='gpt-4-0125-Preview',
        openai_api_base='https://gpt-4-0125-preview-zngd3.openai.azure.com//openai/deployments/gpt-4-0125-Preview/chat/completions?api-version=2024-02-15-preview',
        key='8bc954da91ff4340b63a8c47b57135fc',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        meta_template=api_meta_template,
        query_per_second=0.8,
        run_cfg=dict(num_gpus=1),
        max_out_len=1024, max_seq_len=4096, batch_size=16,retry=1000,temperature=0.2),
]
# GPT3_5 = [
#     dict(abbr='gpt-3.5-turbo-instruct',
#         type=OpenAI, path='gpt-3.5-turbo-instruct',
#         openai_api_base= "https://gpt-35-turbo-instruct-zn.openai.azure.com//openai/deployments/gpt-35-turbo-instruct/completions?api-version=2024-02-15-preview",
#         key='a55c9d72f7ef48d7b637923d6e8815ed',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
#         meta_template=api_meta_template,
#         query_per_second=1,
#         run_cfg=dict(num_gpus=1),
#         max_out_len=1024, max_seq_len=4096, batch_size=8,retry=100,temperature=0.2),
# ]
GPT4o = [
    dict(abbr='gpt-4o',
        type=OpenAI, path='gpt-4o',
        openai_api_base='https://gpt-4o-zngd4.openai.azure.com//openai/deployments/gpt-4o/chat/completions?api-version=2024-06-01',
        key='f9c6bc1a51184aa296954a54fae66f8f',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        meta_template=api_meta_template,
        query_per_second=0.8,
        run_cfg=dict(num_gpus=1),
        max_out_len=1024, max_seq_len=4096, batch_size=16,retry=1000,temperature=0.2),
]

# models=[*GPT4o,*GPT4,*GPT3_5]
models = [*GPT4o,*GPT4]
# models = [*GPT3_5]

# infer = dict(
#     partitioner=dict(type=NaivePartitioner),
#     runner=dict(
#         type=LocalRunner,
#         max_num_workers=4,
#         task=dict(type=OpenICLInferTask)),
# )
