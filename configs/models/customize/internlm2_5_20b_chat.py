from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2_5-20b-chat-hf',
        path='/cpfs01/shared/optimal/model/internlm2_5-20b-chat',
        max_seq_len=4096,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=4),
        stop_words=['</s>', '<|im_end|>'],
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]