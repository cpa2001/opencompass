from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, BM25Retriever, FixedRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.BioBenchmark import BioBenchmarkDataset

bio_reader_cfg = dict(
    input_columns=['question', 'category', 'answer'],
    output_column='answer'
)

bio_infer_cfg_0shot = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="You are a leading expert in {category}. Carefully analyze the following question and provide a step-by-step step solution.\n\nQuestion: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your concise and accurate answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question."
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
# bio_infer_cfg_0shot = dict(
#     prompt_template=dict(
#         type=PromptTemplate,
#         template="你是{category}领域的顶尖专家。请仔细分析以下问题，并提供一个逐步的解决方案。\n\n问题：{question}\n\n请按照以下结构组织你的回答：\n\n分析：\n1. [你推理过程中的第一个关键点或步骤]\n2. [第二个关键点或步骤]\n3. [第三个关键点或步骤]\n(如有必要，可以添加更多步骤)\n\n最终答案：[基于上述分析得出的简洁准确的答案]\n\n请确保你的分析全面透彻，最终答案精确并直接回应问题。"
#     ),
#     retriever=dict(type=ZeroRetriever),
#     inferencer=dict(type=GenInferencer),
# )

bio_infer_cfgs_5shot = {
    'Drug_design': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nQuestion: {question}\nAnswer: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in drug design. Carefully analyze the following question and provide a step-by-step step solution.\n\n</E>\nQuestion: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your concise and accurate answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question.",
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
            template="You are a leading expert in pharmacology. Carefully analyze the following question about drug-drug interactions and provide a step-by-step step solution.\n\n</E>\nQuestion: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your concise and accurate answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question.",
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
            template="You are a leading expert in molecular biology. Carefully analyze the following question about drug-target interactions and provide a step-by-step step solution.\n\n</E>\nQuestion: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer (1 or 0): [Your concise and accurate answer based on the analysis above; output 1 if there is binding affinity, or 0 if there is not]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question.",
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
            template="You are a leading expert in protein biology. Carefully analyze the following sequence and predict its function by providing a step-by-step step solution.\n\n</E>\nSequence: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nPredicted function: [Your concise and accurate prediction based on the analysis above]\n\nEnsure your analysis is thorough and your predicted function is precise and directly addresses the sequence.",
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
            template="You are a leading expert in structural biology. Carefully analyze the following protein secondary structure notation and generate the corresponding amino acid sequence by providing a step-by-step step solution.\n\n</E>\nSecondary Structure: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nGenerated amino acid sequence: [Your generated sequence based on the analysis above]\n\nEnsure your analysis is thorough and your generated sequence is precise and directly addresses the secondary structure.",
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
            template="You are a leading expert in protein structure. Carefully analyze the following amino acid sequence and predict its secondary structure using the sst8 notation by providing a step-by-step step solution.\n\n</E>\nProtein Sequence: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nPredicted secondary structure (sst8 notation): [Your prediction based on the analysis above]\n\nEnsure your analysis is thorough and your predicted structure is precise and directly addresses the sequence.",
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
            template="You are a leading expert in RNA biology. Carefully analyze the following RNA sequence and predict if it can bind to the specified protein by providing a step-by-step step solution.\n\n</E>\nSequence: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nBinding prediction (true/false): [Your prediction based on the analysis above]\n\nEnsure your analysis is thorough and your prediction is precise and directly addresses the sequence.",
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
            template="You are a leading expert in RNA biology. Carefully analyze the following RNA sequence and predict its function by providing a step-by-step step solution.\n\n</E>\nSequence: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nPredicted function: [Your prediction based on the analysis above]\n\nEnsure your analysis is thorough and your predicted function is precise and directly addresses the sequence.",
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
            template="You are a leading expert in RNA structure. Carefully analyze the following dot-bracket notation of the RNA secondary structure and generate the corresponding RNA sequence by providing a step-by-step step solution.\n\n</E>\nSecondary Structure: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nGenerated RNA sequence: [Your generated sequence based on the analysis above]\n\nEnsure your analysis is thorough and your generated sequence is precise and directly addresses the secondary structure.",
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
            template="You are a leading expert in RNA folding. Carefully analyze the following RNA sequence and predict its secondary structure using dot-bracket notation by providing a step-by-step step solution.\n\n</E>\nRNA Sequence: {question}\n\nYour response should be structured as follows:\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nPredicted secondary structure (dot-bracket notation): [Your prediction based on the analysis above]\n\nEnsure your analysis is thorough and your predicted structure is precise and directly addresses the sequence.",
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
            template="You are a leading expert in CRISPR technology. Carefully analyze the following sgRNA sequence and predict its efficiency by providing a step-by-step step solution.\n\n</E>\nSequence: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nPredicted efficiency (%): [Your prediction as a percentage between 0 and 100 based on the analysis above]\n\nEnsure your analysis is thorough and your predicted efficiency is precise and directly addresses the sequence.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'transformed_agentclinic': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nPatient Information: {question}\nDiagnosis: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a medical expert. Analyze the following patient information, physical examination findings, and test results, then provide a diagnosis.\n\n</E>\n\n\nPatient Information: {question}\n\nProvide a detailed analysis and your final diagnosis. Your response should be structured as follows:\n\nAnalysis:\n1. [Key findings from patient history]\n2. [Significant physical examination results]\n3. [Relevant test results]\n4. [Differential diagnosis considerations]\n\nFinal Diagnosis: [Your concise and accurate diagnosis based on the analysis above]\n\nEnsure your analysis is thorough and your final diagnosis is precise and directly addresses the patient's presentation.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    # 'transformed_cmb_clin': dict(
    #     ice_template=dict(
    #         type=PromptTemplate,
    #         template='Example:\nCase: {question}\nAnswer: {answer}\n'
    #     ),
    #     prompt_template=dict(
    #         type=PromptTemplate,
    #         template="You are a medical expert. Read the following patient information and answer the questions.\n\n</E>\n\nQuestion: {question}\n\nProvide detailed answers.",
    #         ice_token='</E>',
    #     ),
    #     retriever=dict(type=BM25Retriever, ice_num=5),
    #     inferencer=dict(type=GenInferencer),
    # ),
    'transformed_cmb_clin': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='参考样例：\n病例：{question}\n答案：{answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="你是一个医学专家。请你阅读病人的病例并回答相应问题。\n\n</E>\n\n病人的病例信息：{question}\n\n请提供详细的解释。",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    # 'transformed_imcs_mrg': dict(
    #     ice_template=dict(
    #         type=PromptTemplate,
    #         template='Example:\nPatient Description and Conversation: {question}\nDiagnostic Report: {answer}\n'
    #     ),
    #     prompt_template=dict(
    #         type=PromptTemplate,
    #         template="You are a medical expert. Based on the following patient description and doctor-patient conversation, generate a diagnostic report.\n\n</E>\n\nQuestion: {question}\n\nProvide the diagnostic report in the following format:\n\n{Provide the report format as per the data}",
    #         ice_token='</E>',
    #     ),
    #     retriever=dict(type=ZeroRetriever),
    #     inferencer=dict(type=GenInferencer),
    # ),
    'transformed_imcs_mrg': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='示例：\n患者自述和医患对话：{question}\n诊疗报告：{answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="您是一位医学专家。根据以下患者自述和医患对话内容，生成对应的诊疗报告。\n\n</E>\n\n问题：{question}\n\n请按照以下格式提供诊疗报告：\n\n{提供报告格式}",
            ice_token='</E>',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    ),
    # 'transformed_cmb_exam': dict(
    #     ice_template=dict(
    #         type=PromptTemplate,
    #         template='Example:\nMedical question: {question}\nCorrect answer: {answer}\n'# Need to be modified here
    #     ),
    #     prompt_template=dict(
    #         type=PromptTemplate,
    #         template="Answer the following multiple-choice question.\n\n</E>\n\nQuestion: {question}\n\nProvide the correct option.",
    #         ice_token='</E>',
    #     ),
    #     retriever=dict(type=BM25Retriever, ice_num=5),
    #     inferencer=dict(type=GenInferencer),
    # ),
    # 'transformed_cmmlu_tcm': dict(
    #     ice_template=dict(
    #         type=PromptTemplate,
    #         template='Example:\nMedical question: {question}\nCorrect answer: {answer}\n'# Need to be modified here
    #     ),
    #     prompt_template=dict(
    #         type=PromptTemplate,
    #         template="Answer the following multiple-choice question related to Traditional Chinese Medicine.\n\n</E>\n\nQuestion: {question}\n\nProvide the correct option.",
    #         ice_token='</E>',
    #     ),
    #     retriever=dict(type=BM25Retriever, ice_num=5),
    #     inferencer=dict(type=GenInferencer),
    # ),
    # 'transformed_mlecqa_tcm': dict(
    #     ice_template=dict(
    #         type=PromptTemplate,
    #         template='Example:\nMedical question: {question}\nCorrect answer: {answer}\n'# Need to be modified here
    #     ),
    #     prompt_template=dict(
    #         type=PromptTemplate,
    #         template="Answer the following multiple-choice question.\n\n</E>\n\n{question}\n\nProvide the correct option.",
    #         ice_token='</E>',
    #     ),
    #     retriever=dict(type=BM25Retriever, ice_num=5),
    #     inferencer=dict(type=GenInferencer),
    # ),
    # 'transformed_tcmsd': dict(
    #     ice_template=dict(
    #         type=PromptTemplate,
    #         template='Example:\nMedical question: {question}\nCorrect answer: {answer}\n'# Need to be modified here
    #     ),
    #     prompt_template=dict(
    #         type=PromptTemplate,
    #         template="Using Traditional Chinese Medicine diagnostics, determine the patient's disease and syndrome based on the following information.\n\n</E>\n\nQuestion: {question}\n\nProvide your answer in the following format:\n\nDisease: \nSyndrome: ",
    #         ice_token='</E>',
    #     ),
    #     retriever=dict(type=BM25Retriever, ice_num=5),
    #     inferencer=dict(type=GenInferencer),
    # ),
    'transformed_cmb_exam': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='示例：\n医学问题：{question}\n正确答案：{answer}\n'  # 需在此处进行修改
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="请回答以下选择题。\n\n</E>\n\n问题：{question}\n\n提供正确的选项。",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),

    'transformed_cmmlu_tcm': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='示例：\n医学问题：{question}\n正确答案：{answer}\n'  # 需在此处进行修改
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="请回答以下关于中医的选择题。\n\n</E>\n\n问题：{question}\n\n提供正确的选项。",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),

    'transformed_mlecqa_tcm': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='示例：\n医学问题：{question}\n正确答案：{answer}\n'  # 需在此处进行修改
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="请回答以下选择题。\n\n</E>\n\n{question}\n\n提供正确的选项。",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),

    'transformed_tcmsd': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='示例：\n医学问题：{question}\n正确答案：{answer}\n'  # 需在此处进行修改
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="使用中医诊断，根据以下信息判断患者的疾病和证候。\n\n</E>\n\n问题：{question}\n\n请按照以下格式提供您的答案：\n\n疾病：\n证候：",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'transformed_headqa': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nMedical Question: {question}\nCorrect Answer: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in medical sciences. Carefully analyze the following multiple-choice question and provide the correct answer along with your reasoning.\n\n</E>\nMedical Question: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your concise and accurate answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'transformed_medlfqa_healthqa': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nHealth Question: {question}\nAnswer: {free_form_answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a medical expert. Carefully analyze the following health-related question and provide a detailed answer.\n\n</E>\nHealth Question: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your detailed answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is comprehensive and directly addresses the question.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'transformed_medlfqa_kqa': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nKnowledge Question: {question}\nAnswer: {free_form_answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a medical expert. Carefully analyze the following knowledge-based question and provide a detailed answer.\n\n</E>\nKnowledge Question: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your detailed answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is comprehensive and directly addresses the question.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'transformed_medlfqa_liveqa': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nLive QA Question: {question}\nAnswer: {free_form_answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a medical expert. Carefully analyze the following live QA question and provide a comprehensive answer.\n\n</E>\nLive QA Question: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your comprehensive answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is detailed and directly addresses the question.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'transformed_medlfqa_medicationqa': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nMedication Question: {question}\nAnswer: {free_form_answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a medical expert. Carefully analyze the following question about medication and provide a detailed answer.\n\n</E>\nMedication Question: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your detailed answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is comprehensive and directly addresses the question.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'transformed_medmcqa': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nMedical Question: {question}\nCorrect Answer: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in medical sciences. Carefully analyze the following multiple-choice question and provide the correct answer along with your reasoning.\n\n</E>\nMedical Question: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your concise and accurate answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    # 'transformed_medqa_cn': dict(
    #     ice_template=dict(
    #         type=PromptTemplate,
    #         template='Example:\nMedical Question (CN): {question}\nCorrect Answer: {answer}\n'
    #     ),
    #     prompt_template=dict(
    #         type=PromptTemplate,
    #         template="You are a leading expert in Chinese medical sciences. Carefully analyze the following multiple-choice question and provide the correct answer along with your reasoning.\n\n</E>\nMedical Question (CN): {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your concise and accurate answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question.",
    #         ice_token='</E>',
    #     ),
    #     retriever=dict(type=BM25Retriever, ice_num=5),
    #     inferencer=dict(type=GenInferencer),
    # ),
    # 'transformed_medqa_tw': dict(
    #     ice_template=dict(
    #         type=PromptTemplate,
    #         template='Example:\nMedical Question (TW): {question}\nCorrect Answer: {answer}\n'
    #     ),
    #     prompt_template=dict(
    #         type=PromptTemplate,
    #         template="You are a leading expert in Taiwanese medical sciences. Carefully analyze the following multiple-choice question and provide the correct answer along with your reasoning.\n\n</E>\nMedical Question (TW): {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your concise and accurate answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question.",
    #         ice_token='</E>',
    #     ),
    #     retriever=dict(type=BM25Retriever, ice_num=5),
    #     inferencer=dict(type=GenInferencer),
    # ),
    'transformed_medqa_cn': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='示例：\n医学问题 (CN)：{question}\n正确答案：{answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="您是一位中国医学领域的顶尖专家。请仔细分析以下选择题，并提供正确答案以及您的推理过程。\n\n</E>\n医学问题 (CN)：{question}\n\n您的回答结构如下：\n\n分析：\n1. [推理过程中的第一关键点或步骤]\n2. [第二关键点或步骤]\n3. [第三关键点或步骤]\n(如有需要，可添加更多步骤)\n\n最终答案：[基于上述分析的简洁准确答案]\n\n确保您的分析详尽无遗，最终答案精确且直接回应问题。",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),

    'transformed_medqa_tw': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='示例：\n医学问题 (TW)：{question}\n正确答案：{answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="您是一位台湾医学领域的顶尖专家。请仔细分析以下选择题，并提供正确答案以及您的推理过程。\n\n</E>\n医学问题 (TW)：{question}\n\n您的回答结构如下：\n\n分析：\n1. [推理过程中的第一关键点或步骤]\n2. [第二关键点或步骤]\n3. [第三关键点或步骤]\n(如有需要，可添加更多步骤)\n\n最终答案：[基于上述分析的简洁准确答案]\n\n确保您的分析详尽无遗，最终答案精确且直接回应问题。",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    'transformed_medqa_us': dict(
        ice_template=dict(
            type=PromptTemplate,
            template='Example:\nMedical Question (US): {question}\nCorrect Answer: {answer}\n'
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template="You are a leading expert in US medical sciences. Carefully analyze the following multiple-choice question and provide the correct answer along with your reasoning.\n\n</E>\nMedical Question (US): {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your concise and accurate answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question.",
            ice_token='</E>',
        ),
        retriever=dict(type=BM25Retriever, ice_num=5),
        inferencer=dict(type=GenInferencer),
    ),
    # 'transformed_mmcu': dict(
    #     ice_template=dict(
    #         type=PromptTemplate,
    #         template='Example:\nMMCU Medical Question: {question}\nCorrect Answer: {answer}\n'
    #     ),
    #     prompt_template=dict(
    #         type=PromptTemplate,
    #         template="You are a leading expert in fundamental medical sciences. Carefully analyze the following multiple-choice question and provide the correct answer along with your reasoning.\n\n</E>\nMMCU Medical Question: {question}\n\nYour response should be structured as follows:\n\nAnalysis:\n1. [First key point or step in your reasoning]\n2. [Second key point or step]\n3. [Third key point or step]\n(Add more steps if necessary)\n\nFinal answer: [Your concise and accurate answer based on the analysis above]\n\nEnsure your analysis is thorough and your final answer is precise and directly addresses the question.",
    #         ice_token='</E>',
    #     ),
    #     retriever=dict(type=BM25Retriever, ice_num=5),
    #     inferencer=dict(type=GenInferencer),
    # ),
    'transformed_mmcu': dict(
    ice_template=dict(
        type=PromptTemplate,
        template='示例：\nMMCU 医学问题：{question}\n正确答案：{answer}\n'
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template="您是一位基础医学领域的顶尖专家。请仔细分析以下选择题，并提供正确答案以及您的推理过程。\n\n</E>\nMMCU 医学问题：{question}\n\n您的回答结构如下：\n\n分析：\n1. [推理过程中的第一关键点或步骤]\n2. [第二关键点或步骤]\n3. [第三关键点或步骤]\n(如有需要，可添加更多步骤)\n\n最终答案：[基于上述分析的简洁准确答案]\n\n确保您的分析详尽无遗，最终答案精确且直接回应问题。",
        ice_token='</E>',
    ),
    retriever=dict(type=BM25Retriever, ice_num=5),
    inferencer=dict(type=GenInferencer),
    ),
}

no_few_shot_datasets = ['Pfam_design', 'Rfam_design']
ten_shot_datasets = ['Pfam_design_10shot', 'Rfam_design_10shot']


bio_benchmark_datasets = []

base_path = './Bio-Benchmark/'

dataset_paths = [
    ('Drug-benchmark', 'Drug_design.json'),
    ('Drug-benchmark', 'Drug-Drug_Interaction.json'),
    ('Drug-benchmark', 'Drug-Target_Interaction.json'),
    
    ('Protein-benchmark', 'Pfam_design.json'),
    ('Protein-benchmark', 'Pfam_design_10shot.json'),
    ('Protein-benchmark', 'Protein_function_prediction.json'),
    ('Protein-benchmark', 'Protein_inverse_folding.json'),
    ('Protein-benchmark', 'Protein_structure_prediction.json'),
    
    ('RBP-benchmark', 'RNA-binding protein.json'),

    ('RNA-benchmark', 'Rfam_design.json'),
    ('RNA-benchmark', 'Rfam_design_10shot.json'),
    ('RNA-benchmark', 'RNA_function_prediction.json'),
    ('RNA-benchmark', 'RNA_inverse_folding.json'),
    ('RNA-benchmark', 'RNA_structure_prediction.json'),
    ('RNA-benchmark', 'sgRNA_efficiency_prediction.json'),
    
    # sequence above
    # ('transformed_ehr', 'transformed_agentclinic.json'),
    # ('transformed_ehr', 'transformed_cmb_clin.json'),
    # ('transformed_ehr', 'transformed_imcs_mrg.json'),

    # ('transformed_tcmqa', 'transformed_cmb_exam.json'),
    # ('transformed_tcmqa', 'transformed_cmmlu_tcm.json'),
    # ('transformed_tcmqa', 'transformed_mlecqa_tcm.json'),
    # ('transformed_tcmqa', 'transformed_tcmsd.json'),

    # ('transformed_medicalqa', 'transformed_headqa.json'),
    # ('transformed_medicalqa', 'transformed_medlfqa_healthqa.json'),
    # ('transformed_medicalqa', 'transformed_medlfqa_kqa.json'),
    # ('transformed_medicalqa', 'transformed_medlfqa_liveqa.json'),
    # ('transformed_medicalqa', 'transformed_medlfqa_medicationqa.json'),
    # ('transformed_medicalqa', 'transformed_medmcqa.json'),
    # ('transformed_medicalqa', 'transformed_medqa_cn.json'),
    # ('transformed_medicalqa', 'transformed_medqa_tw.json'),
    # ('transformed_medicalqa', 'transformed_medqa_us.json'),
    # ('transformed_medicalqa', 'transformed_mmcu.json'),
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
        # pass
    elif dataset_name.endswith('_10shot'):
        base_name = dataset_name
        if base_name in ten_shot_datasets:
            bio_benchmark_datasets.append(
                dict(
                    abbr=f'{base_name}',
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