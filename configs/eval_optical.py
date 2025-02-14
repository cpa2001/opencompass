from mmengine.config import read_base

with read_base():
    from .datasets.OpticalDataset.OpticalDataset_gen import optical_datasets


    # 1GPU
    # from .models.customize.llama3_1_8b_instruct import models as hf_llama3_1_8b_instruct_model
    # from .models.customize.llama3_8b_instruct import models as hf_llama3_8b_instruct_model
    # from .models.customize.qwen2_7b_instruct import models as hf_qwen2_7b_instruct_model

    # from .models.customize.internlm2_5_7b import models as hf_internlm2_5_7b_model
    # from .models.customize.internlm2_5_7b_chat import models as hf_internlm2_5_7b_chat_model
    
    # from .models.customize.internlm2_5_20b_chat import models as hf_internlm2_5_20b_chat_model
    # from .models.customize.phi3_medium_128k_instruct import models as hf_phi3_medium_128k_instruct_model

    # from .models.customize.yi_1_5_34b_chat import models as hf_yi_1_5_34b_chat_model
    # from .models.customize.gemma2_27b_it import models as hf_gemma2_27b_it_model

    # # # from .models.customize.mistral_nemo_instruct import models as hf_mistral_nemo_instruct_model
    # # # from .models.customize.internlm2_7b import models as hf_internlm2_7b_model
    # # # from .models.customize.internlm2_20b import models as hf_internlm2_20b_model
    # # # from .models.customize.internlm2_7b_chat import models as hf_internlm2_7b_chat_model
    # # # from .models.customize.internlm2_20b_chat import models as hf_internlm2_20b_chat_model
    # # # from .models.customize.gemma2_27b import models as hf_gemma2_27b_model

    # 4GPU
    # from .models.customize.internlm2_5_20b import models as hf_internlm2_5_20b_model
    from .models.customize.internlm2_5_20b_optical_sft import models as hf_internlm2_5_20b_optical_sft_model
    from .models.customize.internlm2_5_20b_optical import models as hf_internlm2_5_20b_optical_model
    # from .models.customize.qwen2_5_72b_instruct_vllm import models as vllm_qwen2_5_72b_instruct_model
    # from .models.customize.qwen2_5_72b_instruct_lmdeploy import models as lmdeploy_qwen2_5_72b_instruct_model
    # from .models.customize.qwen2_5_72b_instruct import models as hf_qwen2_5_72b_instruct_model
    # 6GPU
    # from .models.customize.llama3_70b_instruct import models as hf_llama3_70b_instruct_model
    # from .models.customize.OpticalQwen2_72b import models as hf_OpticalQwen2_72b_model
    # from .models.customize.llama3_1_70b_instruct import models as hf_llama3_1_70b_instruct_model
    # from .models.customize.qwen2_72b_instruct import models as hf_qwen2_72b_instruct_model
    # from .models.customize.mistral_large_2_instruct import models as hf_mistral_large_2_instruct_model
    # from .models.customize.OpticalLlama3_70b import models as hf_OpticalLlama3_70b_model

    # 6GPU
    # from .models.customize.qwen1_5_110b import models as hf_qwen1_5_110b_model
    # from .models.customize.mixtral8x22b_instruct import models as hf_mixtral8x22b_instruct_model

    
work_dir = 'outputs/Optical/'
datasets = [*optical_datasets]

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

evaluation_config = dict(
    multi_inferences=1,
)