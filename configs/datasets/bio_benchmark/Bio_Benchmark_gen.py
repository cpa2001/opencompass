from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, BM25Retriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.BioBenchmark import BioBenchmarkDataset

bio_reader_cfg = dict(
    input_columns=['question', 'category', 'answer'],
    output_column='answer'
)

bio_infer_cfg_0shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="You are a leading expert in {category}. Carefully analyze the following question and provide a step-by-step step solution.\n\n</E>\nQuestion: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your concise and accurate answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question."
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

bio_infer_cfgs_5shot = {
    'Drug_design': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nQuestion: {question}\nAnswer: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in drug design. Carefully analyze the following question and provide a step-by-step step solution.\n\n</E>\n</E>Question: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your concise and accurate answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'Drug-Drug_Interaction': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nQuestion: {question}\nAnswer: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in pharmacology. Carefully analyze the following question about drug-drug interactions and provide a step-by-step step solution.\n\n</E>\n</E>Question: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your concise and accurate answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'Drug-Target_Interaction': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nQuestion: {question}\nAnswer: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in molecular biology. Carefully analyze the following question about drug-target interactions and provide a step-by-step step solution.\n\n</E>\n</E>Question: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer (1 or 0): [Your concise and accurate answer based on the analysis above; output 1 if there is binding affinity, or 0 if there is not]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'Protein_function_prediction': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nSequence: {question}\nFunction: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in protein biology. Carefully analyze the following sequence and predict its function by providing a step-by-step step solution.\n\n</E>\n</E>Sequence: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nPredicted function: [Your concise and accurate prediction based on the analysis above]\n\nEnsure your analysis is thorough and your predicted function is precise and directly addresses the sequence.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'Protein_inverse_folding': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nSecondary Structure: {question}\nSequence: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in structural biology. Carefully analyze the following protein secondary structure notation and generate the corresponding amino acid sequence by providing a step-by-step step solution.\n\n</E>\n</E>Secondary Structure: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nGenerated amino acid sequence: [Your generated sequence based on the analysis above]\n\nEnsure your analysis is thorough and your generated sequence is precise and directly addresses the secondary structure.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'Protein_structure_prediction': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nProtein Sequence: {question}\nSecondary Structure: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in protein structure. Carefully analyze the following amino acid sequence and predict its secondary structure using the sst8 notation by providing a step-by-step step solution.\n\n</E>\n</E>Protein Sequence: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nPredicted secondary structure (sst8 notation): [Your prediction based on the analysis above]\n\nEnsure your analysis is thorough and your predicted structure is precise and directly addresses the sequence.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'RNA-binding protein': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nSequence: {question}\nBinding Affinity: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in RNA biology. Carefully analyze the following RNA sequence and predict if it can bind to the specified protein by providing a step-by-step step solution.\n\n</E>\n</E>Sequence: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nBinding prediction (true/false): [Your prediction based on the analysis above]\n\nEnsure your analysis is thorough and your prediction is precise and directly addresses the sequence.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'RNA_function_prediction': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nSequence: {question}\nFunction: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in RNA biology. Carefully analyze the following RNA sequence and predict its function by providing a step-by-step step solution.\n\n</E>\n</E>Sequence: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nPredicted function: [Your prediction based on the analysis above]\n\nEnsure your analysis is thorough and your predicted function is precise and directly addresses the sequence.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'RNA_inverse_folding': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nSecondary Structure: {question}\nRNA Sequence: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in RNA structure. Carefully analyze the following dot-bracket notation of the RNA secondary structure and generate the corresponding RNA sequence by providing a step-by-step step solution.\n\n</E>\n</E>Secondary Structure: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nGenerated RNA sequence: [Your generated sequence based on the analysis above]\n\nEnsure your analysis is thorough and your generated sequence is precise and directly addresses the secondary structure.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'RNA_structure_prediction': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nRNA Sequence: {question}\nSecondary Structure: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in RNA folding. Carefully analyze the following RNA sequence and predict its secondary structure using dot-bracket notation by providing a step-by-step step solution.\n\n</E>\n</E>RNA Sequence: {question}\n\nYour response should be structured as follows:\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nPredicted secondary structure (dot-bracket notation): [Your prediction based on the analysis above]\n\nEnsure your analysis is thorough and your predicted structure is precise and directly addresses the sequence.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'sgRNA_efficiency_prediction': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nSequence: {question}\nEfficiency (%): {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in CRISPR technology. Carefully analyze the following sgRNA sequence and predict its efficiency by providing a step-by-step step solution.\n\n</E>\n</E>Sequence: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nPredicted efficiency (%): [Your prediction as a percentage between 0 and 100 based on the analysis above]\n\nEnsure your analysis is thorough and your predicted efficiency is precise and directly addresses the sequence.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
}

no_few_shot_datasets = ['Pfam_design', 'Rfam_design']

bio_benchmark_datasets = []

base_path = '/cpfs01/user/chenpengan/root-2/opencompass/Bio-Benchmark/'

dataset_paths = [
    ('Drug-benchmark', 'Drug_design.json'),
    ('Drug-benchmark', 'Drug-Drug_Interaction.json'),
    ('Drug-benchmark', 'Drug-Target_Interaction.json'),
    ('Protein-benchmark', 'Pfam_design.json'),
    ('Protein-benchmark', 'Protein_function_prediction.json'),
    ('Protein-benchmark', 'Protein_inverse_folding.json'),
    ('Protein-benchmark', 'Protein_structure_prediction.json'),
    ('RBP-benchmark', 'RNA-binding protein.json'),
    ('RNA-benchmark', 'Rfam_design.json'),
    ('RNA-benchmark', 'RNA_function_prediction.json'),
    ('RNA-benchmark', 'RNA_inverse_folding.json'),
    ('RNA-benchmark', 'RNA_structure_prediction.json'),
    ('RNA-benchmark', 'sgRNA_efficiency_prediction.json'),
]

for folder, filename in dataset_paths:
    dataset_name = filename.split('.')[0]
    if dataset_name in no_few_shot_datasets:
        bio_benchmark_datasets.append(
            dict(
                abbr=f'{dataset_name}_0shot',
                type=BioBenchmarkDataset,
                path=f'{base_path}{folder}/{filename}',
                reader_cfg=bio_reader_cfg,
                infer_cfg=bio_infer_cfg_0shot,
            )
        )
    else:
        bio_benchmark_datasets.append(
            dict(
                abbr=f'{dataset_name}_0shot',
                type=BioBenchmarkDataset,
                path=f'{base_path}{folder}/{filename}',
                reader_cfg=bio_reader_cfg,
                infer_cfg=bio_infer_cfg_0shot,
            ),
        )
        bio_benchmark_datasets.append(
            dict(
                abbr=f'{dataset_name}_5shot',
                type=BioBenchmarkDataset,
                path=f'{base_path}{folder}/{filename}',
                reader_cfg=bio_reader_cfg,
                infer_cfg=bio_infer_cfgs_5shot[dataset_name],
            )
        )