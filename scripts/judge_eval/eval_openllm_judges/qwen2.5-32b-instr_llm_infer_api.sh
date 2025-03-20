MODEL_NAME_OR_PATH=/NAS0/nlp/shenghua/pretrained/Qwen/Qwen2.5-32B-Instruct
SERVED_MODEL_NAME=qwen2_5_32b_instr
HOST=10.10.10.132
PORT=8001
MAX_MODEL_LEN=2048

GPU=4,5
TP=2

BASE_URL=http://$HOST:$PORT/v1
echo LLM: $SERVED_MODEL_NAME 
echo Base_url: $BASE_URL

# run vllm serve
CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL_NAME_OR_PATH \
--host $HOST --port $PORT \
--served-model-name $SERVED_MODEL_NAME \
--tensor-parallel-size $TP \
--max-model-len $MAX_MODEL_LEN \
--disable-log-requests