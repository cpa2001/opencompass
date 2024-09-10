from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-70b-instruct-hf',
        path='/mnt/hwfile/optimal/chenpengan/Optical-Llama3-70b',
        max_out_len=4096,
        batch_size=16,
        run_cfg=dict(num_gpus=6),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        generation_kwargs=dict(
            do_sample=True,
            top_p=1,
            top_k=50,
            temperature=1.0,
        ),
    )
]
