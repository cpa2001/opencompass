from mmengine.config import read_base

with read_base():
    from .datasets.truthfulqa.truthfulqa_gen_cantonese import truthfulqa_cantonese_datasets
    from .datasets.truthfulqa.truthfulqa_gen_english import truthfulqa_english_datasets
    from .datasets.gsm8k.gsm8k_gen_cantonese import gsm8k_cantonese_datasets
    from .datasets.gsm8k.gsm8k_gen_english import gsm8k_english_datasets
    from .datasets.gsm8k.gsm8k_gen_chinese import gsm8k_chinese_datasets
    from .datasets.cmmlu.cmmlu_ppl_test import cmmlu_ppl_datasets
    from .datasets.cmmlu.cmmlu_gen_test import cmmlu_gen_datasets
    from .datasets.cmmlu.cmmlu_yue_gen_csv import cmmlu_yue_gen_datasets
    from .datasets.ARC_c.ARC_c_gen_english import ARC_c_english_datasets
    from .datasets.ARC_c.ARC_c_gen_cantonese import ARC_c_cantonese_datasets
    from .datasets.ARC_c.ARC_c_NLG_cantonese import ARC_c_NLG_cantonese_datasets
    from .datasets.ARC_c.ARC_c_NLG_english import ARC_c_NLG_english_datasets
    from .datasets.cantonese.en_yue import en_yue_datasets
    from .datasets.cantonese.zh_yue import zh_yue_datasets
    from .datasets.wmt19.wmt19_gen import wmt19_datasets
    from .datasets.gsm8k.gsm8k_translation import gsm8k_translate_datasets
    # from .datasets.cantonese.clean_data import clean_yue_dataset

    # aliyun
    # 1GPU
    # from .models.customize.llama3_1_8b_instruct import models as hf_llama3_1_8b_instruct_model
    # from .models.customize.internlm2_5_20b import models as hf_internlm2_5_20b_model
    # from .models.customize.internlm2_5_20b_chat import models as hf_internlm2_5_20b_chat_model
    # from .models.customize.yi_1_5_34b_chat import models as hf_yi_1_5_34b_chat_model

    # 4GPU
    # from .models.customize.mistral_large_2_instruct import models as hf_mistral_large_2_instruct_model
    # from .models.customize.llama3_1_70b_instruct import models as hf_llama3_1_70b_instruct_model
    # from .models.customize.qwen2_72b_instruct import models as hf_qwen2_72b_instruct_model
    from .models.customize.internlm2_5_20b_optical import models as hf_internlm2_5_20b_optical_model

    # S集群
    # 1GPU
    # from .models.customize.llama3_8b_instruct import models as hf_llama3_8b_instruct_model
    # from .models.customize.qwen2_7b_instruct import models as hf_qwen2_7b_instruct_model

    # from .models.customize.internlm2_5_7b import models as hf_internlm2_5_7b_model
    # from .models.customize.internlm2_5_7b_chat import models as hf_internlm2_5_7b_chat_model
    # from .models.customize.phi3_medium_128k_instruct import models as hf_phi3_medium_128k_instruct_model
    from .models.customize.internlm2_5_20b import models as hf_internlm2_5_20b_model
    from .models.customize.internlm2_5_20b_chat import models as hf_internlm2_5_20b_chat_model

    # from .models.customize.yi_1_5_34b_chat import models as hf_yi_1_5_34b_chat_model
    # from .models.customize.gemma2_27b_it import models as hf_gemma2_27b_it_model
    # # # from .models.customize.mistral_nemo_instruct import models as hf_mistral_nemo_instruct_model
    # # # from .models.customize.internlm2_7b import models as hf_internlm2_7b_model
    # # # from .models.customize.internlm2_20b import models as hf_internlm2_20b_model
    # # # from .models.customize.internlm2_7b_chat import models as hf_internlm2_7b_chat_model
    # # # from .models.customize.internlm2_20b_chat import models as hf_internlm2_20b_chat_model
    # # # from .models.customize.gemma2_27b import models as hf_gemma2_27b_model

    # 4GPU
    # from .models.customize.mistral_large_2_instruct import models as hf_mistral_large_2_instruct_model
    from .models.customize.llama3_70b_instruct import models as hf_llama3_70b_instruct_model
    # from .models.customize.llama3_1_70b_instruct import models as hf_llama3_1_70b_instruct_model
    from .models.customize.qwen2_72b_instruct import models as hf_qwen2_72b_instruct_model

    # 6GPU
    # from .models.customize.mixtral8x22b_instruct import models as hf_mixtral8x22b_instruct_model
    # from .models.customize.qwen1_5_110b import models as hf_qwen1_5_110b_model
    
work_dir = 'outputs/wmt19/'

# datasets = [*truthfulqa_cantonese_datasets]
# datasets = [*truthfulqa_english_datasets]
# datasets = [*truthfulqa_english_datasets, *truthfulqa_cantonese_datasets,*gsm8k_cantonese_datasets, *gsm8k_english_datasets]
# datasets = [*gsm8k_cantonese_datasets]
# datasets = [*gsm8k_english_datasets]
# datasets = [*cmmlu_ppl_datasets]
# datasets = [*cmmlu_gen_datasets]
# datasets = [*ARC_c_cantonese_datasets]
# datasets = [*ARC_c_english_datasets]
# datasets = [*ARC_c_NLG_cantonese_datasets]
# datasets = [*ARC_c_NLG_english_datasets]
# datasets = [*ARC_c_NLG_cantonese_datasets,*ARC_c_NLG_english_datasets]
# datasets = [*cmmlu_yue_gen_datasets, *cmmlu_gen_datasets]
# datasets = [*cmmlu_yue_gen_datasets]
# datasets = [*en_yue_datasets]
# datasets = [*zh_yue_datasets]
# datasets = [*en_yue_datasets, *zh_yue_datasets]
# datasets = [*clean_yue_dataset]
# datasets = [*gsm8k_chinese_datasets]
datasets = [*wmt19_datasets]
# datasets = [*gsm8k_translate_datasets]
<<<<<<< Updated upstream
# datasets= [*gsm8k_translate_train_datasets]
=======
>>>>>>> Stashed changes
# models = hf_llama3_8b_instruct_model
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
