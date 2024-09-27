from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='yi-1.5-34b-chat-hf',
        path='/mnt/hwfile/optimal/chenpengan/yi-1_5-34b-chat',
        max_seq_len=4096,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=6),
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]
