MODEL_NAME_OR_PATH=/NAS0/nlp/shenghua/pretrained/Qwen/QwQ-32B
SERVED_MODEL_NAME=qwq_32b
HOST=10.10.10.121
PORT=8000
TP=0
CUDA_VISIBLE_DEVICES=6 vllm serve $MODEL_NAME_OR_PATH \
--host $HOST --port $PORT \
--served-model-name $SERVED_MODEL_NAME \
--tensor-parallel-size $TP \
--uvicorn-log-level warning
