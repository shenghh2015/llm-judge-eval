from simple_eval.judge import LLMJudge
from utils.utils_read_write import jsonl_file_read
import os

if __name__ == "__main__":

  # openai model: gpt-4o, gpt-to-mini
  # model = "gpt-4o"
  # model = "gpt-4o-mini"
  # base_url = ""
  # api_key = os.environ.get("OPENAI_API_KEY")

  # opensource model served by vllm
  # model = "qwen2_5_32b_instr"
  # base_url = "http://10.10.10.132:8001/v1"
  # model = "qwq_32b"
  # base_url = "http://10.10.10.121:8000/v1"

  model = "deepseek_r1_32b"
  base_url = "http://10.10.10.121:8001/v1"
  api_key = "EMPTY"

  PROMPT_TEMPLATE = """
    Which of the following summaries does a better job of summarizing the most
    important points in the given forum post, without including unimportant or
    irrelevant details? A good summary is both precise and concise.
    
    Please answer the question in a json format, which contains two keys: "comparison" and "preferred".
    The "comparision" key should contain a one-sentence comparison of the two summaries, explaining which
    you prefer and why. 
    The "preferred" key should contain a binary value of "1" or "0", where "1" indicates that
    you prefer Summary A and "0" indicates that you don't prefer Summary A.
    
    The post and the two summaries are as follows:
    Post:
    {}
    Summary A: {}
    Summary B: {}
    """

  llm_judge = LLMJudge(base_url=base_url,
                       api_key=api_key,
                       model=model,
                       prompt_template=PROMPT_TEMPLATE,
                       max_new_tokens=1024)

  # load dataset
  datalist = list(
      jsonl_file_read(
          "./example_datasets/sampled_data.summarize.2025-03-18/data.split_0.jsonl"
      ))

  # make llm judging
  juding_results = llm_judge.make_judging(datalist, temperature=0.1)
  print(juding_results[0]["judging_text"])  # print the first judging result
  print(
      juding_results[0]["judging_text_swapped"]
  )  # print the juding result after swapping the powsition of two responses

  # llm judge evaluation
  eval_result = llm_judge.evaluate()
  print(eval_result)
  os.makedirs("./outputs/.cache_eval_results", exist_ok=True)
  with open(f"./outputs/.cache_eval_results/{model}_summary_eval_result.jsonl",
            "w") as f:
    f.write(str(eval_result))
