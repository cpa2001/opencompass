from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='mixtral-large-2-instruct-hf',
        path='/cpfs01/shared/optimal/model/mistral-large-instruct-2407',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]
