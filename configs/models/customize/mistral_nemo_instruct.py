from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='mixtral-nemo-instruct-hf',
        path='/mnt/hwfile/optimal/chenpengan/mistral-nemo-instruct-2407',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        # generation_kwargs=dict(
        #     do_sample=True,
        #     top_p=1,
        #     temperature=0.2,
        # ),
    )
]
