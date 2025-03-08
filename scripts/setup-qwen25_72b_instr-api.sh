MODEL_NAME_OR_PATH=/NAS0/nlp/shenghua/pretrained/Qwen/Qwen2.5-72B-Instruct
SERVED_MODEL_NAME=qwen25_72b_instr
HOST=10.10.10.131
PORT=8000
TP=4
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $MODEL_NAME_OR_PATH \
--host $HOST --port $PORT \
--served-model-name $SERVED_MODEL_NAME \
--tensor-parallel-size $TP \
--uvicorn-log-level warning
