from openai import OpenAI
from threading import Thread
from utils.utils_read_write import jsonl_file_write
from simple_eval.eval_utils import (prompting, call_model_service,
                                    extract_json_from_text, compute_acc_both,
                                    compute_acc_random)

import os

class LLMJudge:

  def __init__(self, base_url, api_key, model, prompt_template,
               max_new_tokens):

    self.base_url = base_url
    self.api_key = api_key
    self.model = model
    self.prompt_template = prompt_template
    self.max_new_tokens = max_new_tokens
    self.model_client = OpenAI(api_key=api_key, base_url=base_url)
    
  def _make_judge_for_response_pair(self, temperature, data, idx, results):
    user_input = prompting(self.prompt_template, data["prompt"],
                           data["chosen"], data["rejected"])
    user_input_reversed = prompting(self.prompt_template, data["prompt"],
                                    data["rejected"], data["chosen"])
    data["judging_text"] = call_model_service(self.model_client, self.model,
                                              user_input, temperature,
                                              self.max_new_tokens)
    data["judging_text_swapped"] = call_model_service(self.model_client,
                                                      self.model,
                                                      user_input_reversed,
                                                      temperature,
                                                      self.max_new_tokens)
    results[idx] = data

  def make_judging(self, datalist, temperature):
    result_list = [None] * len(datalist)
    threads = []
    for idx, data in enumerate(datalist):
      threads.append(
          Thread(target=self._make_judge_for_response_pair,
                 args=(temperature, data, idx, result_list)))
      threads[-1].start()
    for td in threads:
      td.join()
    self.juding_results = result_list
    return result_list

  def _binarize_juding_results(self, result_list):
    position_unswapped = [
        int(
            extract_json_from_text(result["judging_text"]).get(
                "preferred", "-1")) for result in result_list
    ]
    position_swapped = [
        int(
            extract_json_from_text(result["judging_text_swapped"]).get(
                "preferred", "-1")) for result in result_list
    ]
    return position_unswapped, position_swapped

  def evaluate(self, random_seed=42):

    unswapped, swapped = self._binarize_juding_results(self.juding_results)

    # Accuracy (Both)
    acc_both, valid_count_both = compute_acc_both(unswapped, swapped)

    # Accuracy (Random)
    acc_random, valid_count_random = compute_acc_random(
        unswapped, swapped, random_seed)

    return {
        "model": self.model,
        "acc_both": round(acc_both, 3),
        "valid_count_both": valid_count_both,
        "acc_random": round(acc_random, 3),
        "valid_count_random": valid_count_random
    }

  def save(self, output_path):
    output_dir = os.path.basename(output_path)
    os.makedirs(output_dir, exist_ok=True)
    jsonl_file_write(self.juding_results, output_path)