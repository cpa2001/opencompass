from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='internlm2_5-20b-optical-hf',
        path='/cpfs01/user/chenpengan/root-2/xtuner/sft_model/hf-model-sft-optical-5epoch',
        max_seq_len=4096,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=4),
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.8,
        ),
    )
]
