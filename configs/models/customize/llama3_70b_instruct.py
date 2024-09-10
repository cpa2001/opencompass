from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-70b-instruct-hf',
        path='/mnt/hwfile/optimal/chenpengan/llama3-70b-instruct',
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=4),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            temperature=0.2,
        ),
    )
]
