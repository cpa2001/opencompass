from opencompass.models import OpenAISDK

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        type=OpenAISDK,
        abbr='llama-3.1-405b-vllm',
        path='/cpfs01/shared/optimal/model/llama3_1-405b-instruct-fp8',
        max_out_len=2048,  # Adjust based on your use case
        batch_size=4,  # Adjust for batch size based on your resources
        run_cfg=dict(
            num_gpus=8,  # Adjust depending on your available GPUs
            tensor_parallel_size=8,
            dtype='auto',
            host='0.0.0.0',
            port=8000,
        ),
    ),
]
