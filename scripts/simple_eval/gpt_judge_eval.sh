## 1. Set your OpenAI API key
export OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]

## 2. Run the evaluation scripts
cd llm-judge-eval
python simple_eval/judge_eval.py --model gpt-4o-mini --task summary            # summary task
python simple_eval/judge_eval.py --model gpt-4o-mini --task hh_rlhf_helpful    # hh_rlhf_helpful task
python simple_eval/judge_eval.py --model gpt-4o --task summary                 # summary task
python simple_eval/judge_eval.py --model gpt-4o --task summhh_rlhf_helpfulary  # summary task