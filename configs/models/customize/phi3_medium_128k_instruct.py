from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='phi-3-medium-128k-instruct-hf',
        path='/mnt/hwfile/optimal/chenpengan/phi3-medium-128k-instruct',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]