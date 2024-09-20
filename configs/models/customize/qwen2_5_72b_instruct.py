from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-72b-instruct-hf',
        path='/cpfs01/shared/optimal/model/qwen2_5-72b-instruct',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]