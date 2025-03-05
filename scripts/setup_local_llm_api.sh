MODEL_NAME_OR_PATH=/NAS0/nlp/shenghua/pretrained/Qwen/Qwen2.5-7B-Instruct
SERVED_MODEL_NAME=qwen25-7b-instr
HOST=10.10.10.130
PORT=8001
CUDA_VISIBLE_DEVICES=7 vllm serve $MODEL_NAME_OR_PATH \
--host $HOST --port $PORT \
--served-model-name $SERVED_MODEL_NAME
