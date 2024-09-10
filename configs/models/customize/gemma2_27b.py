from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='gemma2-27b-hf',
        path='/mnt/hwfile/optimal/chenpengan/gemma2-27b',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16',
        ),
    )
]