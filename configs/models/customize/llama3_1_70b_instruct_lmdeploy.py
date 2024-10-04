from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='llama-3_1-70b-instruct-turbomind',
        path='/cpfs01/shared/optimal/model/llama3_1-70b-instruct',
        engine_config=dict(max_batch_size=32, tp=8),
        gen_config=dict(top_k=1, temperature=0.2, top_p=0.999, max_new_tokens=2048),
        max_seq_len=4096,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=8),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]