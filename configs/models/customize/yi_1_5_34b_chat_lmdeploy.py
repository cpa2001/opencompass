from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='yi-1.5-34b-chat-hf-turbomind',
        path='/cpfs01/shared/optimal/model/yi-1_5-34b-chat',
        engine_config=dict(session_len=4096, max_batch_size=256, tp=8),
        gen_config=dict(top_k=1, temperature=0.2, top_p=0.999, max_new_tokens=2048),
        max_seq_len=4096,
        max_out_len=2048,
        batch_size=256,
        run_cfg=dict(num_gpus=8),
    )
]