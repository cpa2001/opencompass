from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, SlidingWindowRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator,MultiInferenceConsistencyEvaluator
from opencompass.datasets import BaseDataset
from opencompass.utils.text_postprocessors import first_option_postprocess,first_capital_postprocess
from opencompass.datasets import OpticalDataset

optical_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

optical_infer_cfg_0shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="You are an expert in optical physics. Your task is to answer the following multiple-choice question by thinking step-by-step. Answer with the option's letter from the given choices directly. Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"
                )
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, num_inferences=1),
)

optical_infer_cfg_5shot = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='Example: \nQuestion: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: {answer}\n'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template="</E>\nYou are an expert in optical physics. Your task is to answer the following multiple-choice question by thinking step-by-step. Answer with the option's letter from the given choices directly. Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: ",
        ice_token='</E>',
    ),
    retriever=dict(type=SlidingWindowRetriever, k=5),
    inferencer=dict(type=GenInferencer, num_inferences=1),
)

optical_eval_cfg = dict(
    # evaluator=dict(type=MultiInferenceConsistencyEvaluator, num_inferences=1),
    evaluator=dict(type=AccwithDetailsEvaluator),
    # pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    pred_postprocessor=dict(type=first_capital_postprocess)
)

optical_datasets = []

# for dataset_name in ['ethic', 'exam_chinese', 'exam', 'hard_book', 'laser_safety', 'optics_book', 'photonics_book_question']:
for dataset_name in ['exam_chinese', 'exam', 'hard_book', 'laser_safety', 'optics_book', 'ethic','photonics_book']:
# for dataset_name in ['2-ANSI']:
    # 0-shot configuration
    optical_datasets.append(
        dict(
            abbr=f'{dataset_name}_0shot',
            type=OpticalDataset,
            path=f'/mnt/petrelfs/chenpengan/opencompass/OpticalDataset/{dataset_name}_test.csv',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='test',
                test_split='test'
            ),
            infer_cfg=optical_infer_cfg_0shot,
            eval_cfg=optical_eval_cfg,
        )
    )
    
    # 5-shot configuration
    optical_datasets.append(
        dict(
            abbr=f'{dataset_name}_5shot',
            type=OpticalDataset,
            path=f'/mnt/petrelfs/chenpengan/opencompass/OpticalDataset/{dataset_name}_test.csv',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='test',
                test_split='test'
            ),
            infer_cfg=optical_infer_cfg_5shot,
            eval_cfg=optical_eval_cfg,
        )
    )