from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen2-7b-hf',
        path='/mnt/hwfile/optimal/chenpengan/qwen2-7b-instruct',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]
