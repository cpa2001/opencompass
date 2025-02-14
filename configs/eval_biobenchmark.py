from mmengine.config import read_base

with read_base():
    from .datasets.bio_benchmark.Bio_Benchmark_gen import bio_benchmark_datasets


    # aliyun
    # 1GPU
    from .models.customize.llama3_1_8b_instruct import models as hf_llama3_1_8b_instruct_model
    # from .models.customize.internlm2_5_20b import models as hf_internlm2_5_20b_model
    # from .models.customize.yi_1_5_34b_chat import models as hf_yi_1_5_34b_chat_model
    # from .models.customize.internlm2_5_20b_lmdeploy import models as lmdeploy_internlm2_5_20b_model

    # 4GPU-vllm
    # from .models.customize.qwen2_5_72b_instruct_vllm import models as vllm_qwen2_5_72b_instruct_model
    # from .models.customize.mistral_large_2_instruct_vllm import models as vllm_mistral_large_2_instruct_model
    # # from .models.customize.qwen2_5_72b_instruct import models as hf_qwen2_5_72b_instruct_model
    # from .models.customize.llama3_1_70b_instruct import models as hf_llama3_1_70b_instruct_model
    # # from .models.customize.mistral_large_2_instruct import models as hf_mistral_large_2_instruct_model
    # # from .models.customize.mistral_large_2_instruct_lmdeploy import models as lmdeploy_mistral_large_2_instruct_model
    # # from .models.customize.llama3_1_405b_fp8_instruct_vllm_openai import models as vllm_llama3_1_405b_instruct_model
    # # from .models.customize.qwen2_5_72b_instruct_lmdeploy import models as lmdeploy_qwen2_5_72b_instruct_model
    # from .models.customize.llama3_1_405b_fp8_instruct import models as vllm_llama3_1_405b_fp8_instruct_model

    
work_dir = 'outputs/bio-benchmark/'

datasets = [*bio_benchmark_datasets]

# models = hf_llama3_8b_instruct_model
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
