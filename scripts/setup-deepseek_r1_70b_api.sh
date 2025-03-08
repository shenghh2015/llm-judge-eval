MODEL_NAME_OR_PATH=/NAS0/nlp/shenghua/pretrained/deepseek-ai/DeepSeek-R1-Distill-Llama-70B
SERVED_MODEL_NAME=deepseek_r1_70b
MAX_MODEL_LEN=2048
HOST=10.10.10.131
PORT=8001
TP=4
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve $MODEL_NAME_OR_PATH \
--host $HOST --port $PORT \
--served-model-name $SERVED_MODEL_NAME \
--tensor-parallel-size $TP \
--max-model-len $MAX_MODEL_LEN \
--uvicorn-log-level warning
