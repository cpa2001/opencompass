from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3_1-70b-instruct-hf',
        path='/mnt/hwfile/optimal/chenpengan/llama3_1-70b-instruct',
        max_seq_len=4096,
        max_out_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=6),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]
