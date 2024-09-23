from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen2.5-72b-instruct-turbomind',
        path='/cpfs01/shared/optimal/model/qwen2_5-72b-instruct',
        engine_config=dict(session_len=16384, max_batch_size=16, tp=4),
        gen_config=dict(top_k=1, temperature=0.2, top_p=0.9, max_new_tokens=4096),
        max_seq_len=16384,
        max_out_len=4096,
        batch_size=32,
        run_cfg=dict(num_gpus=8),
    )
]