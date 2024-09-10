from opencompass.models import HuggingFaceBaseModel


models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='internlm2_5-7b-hf',
        path='/mnt/hwfile/optimal/chenpengan/internlm2_5-7b',
        max_seq_len=4096,
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