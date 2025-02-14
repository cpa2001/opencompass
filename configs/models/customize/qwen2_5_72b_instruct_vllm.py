from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2_5-72b-instruct-vllm',
        path='/cpfs01/shared/optimal/model/qwen2_5-72b-instruct',
        model_kwargs=dict(tensor_parallel_size=8),
        max_out_len=4096,
        batch_size=128,
        generation_kwargs=dict(temperature=0.2),
        run_cfg=dict(num_gpus=8),
    )
]