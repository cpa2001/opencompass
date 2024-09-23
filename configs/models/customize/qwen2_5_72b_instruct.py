from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-72b-instruct-hf',
        path='/cpfs01/shared/optimal/model/qwen2_5-72b-instruct',
        max_out_len=1024,
        batch_size=32,
        run_cfg=dict(num_gpus=8),
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]