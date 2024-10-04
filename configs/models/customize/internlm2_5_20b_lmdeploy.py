from opencompass.models import TurboMindModel

models = [
    dict(
        type=TurboMindModel,
        abbr='internlm2_5-20b-turbomind',
        path='/cpfs01/shared/optimal/model/internlm2_5-20b',
        engine_config=dict(session_len=4096, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=0.2, top_p=0.999, max_new_tokens=1024),
        max_seq_len=4096,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]