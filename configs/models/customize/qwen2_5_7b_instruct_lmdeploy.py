from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen2.5-7b-instruct-turbomind',
        path='/cpfs01/shared/optimal/model/qwen2_5-7b-instruct',
        engine_config=dict(session_len=4096, max_batch_size=2048, tp=4),
        gen_config=dict(top_k=1, temperature=0.2, top_p=0.999, max_new_tokens=2048),
        max_seq_len=4096,
        max_out_len=2048,
        batch_size=512,
        run_cfg=dict(num_gpus=4),
    )
]