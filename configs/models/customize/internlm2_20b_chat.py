from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2-chat-20b-hf',
        path='/mnt/hwfile/optimal/chenpengan/internlm2-20b-chat',
        max_seq_len=4096,
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
        stop_words=['</s>', '<|im_end|>'],
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]
