vllm serve /cpfs01/shared/optimal/model/llama3_1-405b-instruct-fp8 \
     --dtype auto \
     --host 0.0.0.0 \
     --port 8000 \
     --tensor-parallel-size 8