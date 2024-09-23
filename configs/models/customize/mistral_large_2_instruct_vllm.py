from opencompass.models import VLLMwithChatTemplate


models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='mixtral-large-2-instruct-vllm',
        path='/cpfs01/shared/optimal/model/mistral-large-instruct-2407',
        model_kwargs=dict(tensor_parallel_size=8),
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=8),
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]