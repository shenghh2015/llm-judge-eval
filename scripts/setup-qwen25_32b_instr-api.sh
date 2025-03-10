MODEL_NAME_OR_PATH=/NAS0/nlp/shenghua/pretrained/Qwen/Qwen2.5-32B-Instruct
SERVED_MODEL_NAME=qwen25_32b_instr
HOST=10.10.10.130
PORT=8004
TP=2
CUDA_VISIBLE_DEVICES=1,7 vllm serve $MODEL_NAME_OR_PATH \
--host $HOST --port $PORT \
--served-model-name $SERVED_MODEL_NAME \
--tensor-parallel-size $TP \
--uvicorn-log-level warning
