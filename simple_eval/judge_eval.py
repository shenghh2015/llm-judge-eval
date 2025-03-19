from simple_eval.judge import LLMJudge
from simple_eval.templates import PROMPT_TEMPLATES
from utils.utils_read_write import jsonl_file_read
import os
import argparse


def get_args():
  parser = argparse.ArgumentParser(description="LLM Judge Evaluation")
  parser.add_argument("--model",
                      type=str,
                      required=True,
                      help="LLM model name")
  parser.add_argument("--base_url",
                      type=str,
                      default="",
                      help="LLM service base URL")
  parser.add_argument("--api_key",
                      type=str,
                      default="EMPTY",
                      help="API key for LLM service")
  parser.add_argument("--task",
                      type=str,
                      default="summary",
                      help="Task type: summary, hh_rlhf_helpful, or general")
  return parser.parse_args()


if __name__ == "__main__":

  args = get_args()

  model = args.model
  base_url = args.base_url
  api_key = args.api_key
  task = args.task

  prompt_template = PROMPT_TEMPLATES[task]

  if model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4"]:
    api_key = os.environ.get("OPENAI_API_KEY", "[YOUR API KEY]")

  llm_judge = LLMJudge(base_url=base_url,
                       api_key=api_key,
                       model=model,
                       prompt_template=prompt_template,
                       max_new_tokens=1024)

  # load eval dataset
  if task == "summary":
    datalist = list(
        jsonl_file_read(
            "./examples/example_datasets/sampled_data.summarize.2025-03-18/data.split_0.jsonl"
        ))
  elif task == "hh_rlhf_helpful":
    datalist = list(
        jsonl_file_read(
            "./examples/example_datasets/sampled_data.hhrlhf_helpful.2025-03-18/data.split_0.jsonl"
        ))

  # make judging results
  juding_results = llm_judge.make_judging(datalist, temperature=0.1)
  print(juding_results[0]["judging_text"])  # print the first judging result
  print(
      juding_results[0]["judging_text_swapped"]
  )  # print the juding result after swapping the powsition of two responses

  # llm judge evaluation
  eval_result = llm_judge.evaluate()
  print(eval_result)

  # save eval result
  os.makedirs("./outputs/.cache_eval_results", exist_ok=True)
  with open(f"./outputs/.cache_eval_results/{model}_{task}_eval_result.jsonl",
            "w") as f:
    f.write(str(eval_result))
