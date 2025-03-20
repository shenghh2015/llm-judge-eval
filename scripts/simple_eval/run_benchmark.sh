# LLM_SERVERS = {
#     "deepseek_r1_70b": "http://10.10.10.128:8000/v1",
#     "deepseek_r1_32b": "http://10.10.10.121:8001/v1",
#     "qwq_32b": "http://10.10.10.121:8000/v1",
#     "qwen2_5_32b_instr": "http://10.10.10.121:8002/v1",
#     "qwen2_5_72b_instr": "http://10.10.10.128:8001/v1",
#     "llama3_1_70b_instr": "http://10.10.10.131:8000/v1",
#     "llama3_3_70b_instr": "http://10.10.10.131:8001/v1",
#     "gpt-4o": "https://api.openai.com/v1",
#     "gpt-4o-mini": "https://api.openai.com/v1",
#     "gpt-3.5-turbo": "https://api.openai.com/v1"
# }
# python simple_eval/judge_eval.py --model deepseek_r1_70b --task summary &
python simple_eval/judge_eval.py --model deepseek_r1_32b --task summary &
python simple_eval/judge_eval.py --model qwq_32b --task summary &
# python simple_eval/judge_eval.py --model qwen2_5_32b_instr --task summary &
python simple_eval/judge_eval.py --model qwen2_5_72b_instr --task summary &
python simple_eval/judge_eval.py --model llama3_1_70b_instr --task summary &
python simple_eval/judge_eval.py --model llama3_3_70b_instr --task summary &

python simple_eval/judge_eval.py --model deepseek_r1_70b --task hh_rlhf_helpful &
python simple_eval/judge_eval.py --model deepseek_r1_32b --task hh_rlhf_helpful &
python simple_eval/judge_eval.py --model qwq_32b --task hh_rlhf_helpful &
python simple_eval/judge_eval.py --model qwen2_5_32b_instr --task hh_rlhf_helpful &
python simple_eval/judge_eval.py --model qwen2_5_72b_instr --task hh_rlhf_helpful &
python simple_eval/judge_eval.py --model llama3_1_70b_instr --task hh_rlhf_helpful &
python simple_eval/judge_eval.py --model llama3_3_70b_instr --task hh_rlhf_helpful &

wait