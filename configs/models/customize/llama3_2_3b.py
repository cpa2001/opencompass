from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-3.2-3b-hf',
        path='/mnt/hwfile/optimal/chenpengan/llama3_2-3b',
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]