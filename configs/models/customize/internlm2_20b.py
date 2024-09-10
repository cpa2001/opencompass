from opencompass.models import HuggingFaceBaseModel


models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='internlm2-20b-hf',
        path='/mnt/hwfile/optimal/chenpengan/internlm2-20b',
        max_seq_len=4096,
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]
