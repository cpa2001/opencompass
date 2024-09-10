from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='gemma2-27b-it-hf',
        path='/mnt/hwfile/optimal/chenpengan/gemma2-27b-it',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<end_of_turn>'],
        model_kwargs=dict(
            torch_dtype='torch.bfloat16',
        )
    )
]