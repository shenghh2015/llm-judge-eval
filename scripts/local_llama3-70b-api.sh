MODEL_NAME_OR_PATH=/NAS0/nlp/shenghua/pretrained/meta-llama/Meta-Llama-3-70B-Instruct
SERVED_MODEL_NAME=llama3_70b_instr
HOST=10.10.10.130
PORT=8005
TP=4
CUDA_VISIBLE_DEVICES=2,3,4,5 vllm serve $MODEL_NAME_OR_PATH \
--host $HOST --port $PORT \
--served-model-name $SERVED_MODEL_NAME \
--tensor-parallel-size $TP
