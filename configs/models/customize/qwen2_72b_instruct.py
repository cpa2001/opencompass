from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen2-72b-hf',
        path='/cpfs01/shared/optimal/model/qwen2-72b-instruct',
        max_out_len=1024,
        batch_size=64,
        run_cfg=dict(num_gpus=8),
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]
