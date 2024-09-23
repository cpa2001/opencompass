from opencompass.models import HuggingFaceBaseModel


models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='internlm2_5-20b-hf',
        path='/cpfs01/shared/optimal/model/internlm2_5-20b',
        max_seq_len=4096,
        max_out_len=1024,
<<<<<<< Updated upstream
        batch_size=16,
=======
        batch_size=8,
>>>>>>> Stashed changes
        run_cfg=dict(num_gpus=4),
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]