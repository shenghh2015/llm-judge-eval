## 1. Set up llm inference API using vllm
CUDA_VISIBLE_GPUS=0 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --served-model-name qwen2.5_7b_instruct --max-model-len 2024

## 2. Check if the API is running
curl http://localhost:8000/v1
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5_7b_instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'

## 3. Run the evaluation script
cd llm-judge-eval
# summary task
python simple_eval/judge_eval.py --model qwen2.5_7b_instruct --base_url http://localhost:8000/v1 --task summary
# hh_rlhf_helpful task    
python simple_eval/judge_eval.py --model qwen2.5_7b_instruct --base_url http://localhost:8000/v1 --task hh_rlhf_helpful 