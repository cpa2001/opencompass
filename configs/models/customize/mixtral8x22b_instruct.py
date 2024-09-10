from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='mixtral-8x22b-instruct-v0.1-hf',
        path='/mnt/hwfile/optimal/chenpengan/mixtral-8x22b-instruct',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=6),
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]
