"""
Metrics Computation (Self-Consistency, Accuracy and Biases).    
"""
from judge_eval.utils.others import new_dir, find_latest
from judge_eval.utils.read_write import jsonl_file_read, jsonl_file_write
from judge_eval.utils.metrics.accuracy_bias import analyze_separate_rewarding_accuracy_bias
from judge_eval.utils.metrics.accuracy_bias import format_method_comparison_results
from judge_eval.utils.metrics.self_consistency import evaluate_self_consistency
from judge_eval.utils.metrics.post_process import aggregate_results, split_results, get_rank
from datetime import date
import pandas as pd
import os
import regex


class MetricsComputation:

  def __init__(self, data_path, num_runs, cache_dir):
    """
    Args:
      - data_path: data path to the judging results.
      - num_runs: number of runs to compute self-consistency. 
      - cache_dir: cached dir to store metrics computation results.
    """
    self.data_path = data_path
    self.num_runs = num_runs
    self.cache_dir = cache_dir

  def make_judge_name(self, llm, template_name):
    '''Make judge_name based on llm and template.'''
    # Get the judge name to create the cached path.
    llm_name = regex.sub(r'[^a-zA-Z0-9]', '', llm)
    template_name = template_name.split('_')[0]  # remove the task
    template_name = regex.sub(r'[^a-zA-Z0-9]', '', template_name)
    return f"{llm_name}_{template_name}"  # judge name

  def _check_cached_file(self, path_format):
    '''Check if cached file exists.'''
    latest = find_latest(path_format)
    # Level 0: data path.
    # If multiple sampled version, the latest one will be returned.
    if latest == '': return False
    # Level 1: meta data info, such as size.
    if os.path.getsize(latest) == 0: return False
    return True

  def _check_judge_results(self, dataset_id, judge_name, split_id, num_runs):
    '''Check the judge result files to compute the self-consistency results.'''
    is_valid = True
    filenames = []
    for run_id in range(num_runs):
      result_path = os.path.join(self.data_path, dataset_id, "judging_results",
                                 judge_name)
      result_path = os.path.join(
          result_path,
          f"**.{dataset_id}.{judge_name}.split_id{split_id}.run_id{run_id}.jsonl"
      )
      is_valid = is_valid and self._check_cached_file(result_path)
      filenames.append(find_latest(result_path))
    return is_valid, filenames

  def _check_cached_acc_bias_results(self, path_format, num_splits, num_eval):
    '''Check if cached results for accuracy and bias exist.'''
    latest = find_latest(path_format)
    # Level 0: data path.
    # If multiple sampled version, the latest one will be returned.
    if latest == '': return False
    # Level 1: meta data info, such as size.
    if os.path.getsize(latest) == 0: return False
    # Level 2: data completeness.
    result_splits = [result for result in jsonl_file_read(latest)][0]
    num_splits_from_result = len(
        result_splits["Valid Count"].split('|')[0].split(','))
    if num_splits_from_result != num_splits: return False
    if result_splits["num_eval"] != num_eval: return False
    return True

  def _check_cached_self_consistency_results(self, path_format, split_id,
                                             num_eval, num_runs):
    '''Check if cached results for self-consistency exists.'''
    latest = find_latest(path_format)
    # Level 0: data path.
    # If multiple sampled version, the latest one will be returned.
    if latest == '': return False
    # Level 1: meta data info, such as size.
    if os.path.getsize(latest) == 0: return False
    # Level 2: if key fields match
    self_consist_results = [result for result in jsonl_file_read(latest)][0]
    if self_consist_results["split_id"] != split_id: return False
    if self_consist_results["num_runs"] != num_runs: return False
    if self_consist_results["num_eval"] != num_eval: return False
    return True

  def _check_post_processed_results(self, dataset_id, llms, templates,
                                    num_eval, num_splits, num_runs, split_id):
    '''Check the post processed results.'''
    latest = find_latest(
        os.path.join(self.cache_dir, dataset_id, "all_results", "**"))
    # Level 0: Check the folder
    if latest == '': return False
    # Level 1: Check the files
    check_files = [
        f"all_results.{dataset_id}.jsonl", f"rank_results.{dataset_id}.jsonl"
    ]
    for sdx in range(num_splits):
      check_files.append(f"split{sdx}_results.{dataset_id}.jsonl")
    is_contain_all = True
    for file in check_files:
      is_contain_all = is_contain_all and os.path.exists(
          os.path.join(latest, file))
    if not is_contain_all:
      return False
    # Level 2: Check the key fields (llm, template, num_eval, num_splits, num_runs, split_id)
    all_results = jsonl_file_read(
        os.path.join(latest, f"all_results.{dataset_id}.jsonl"))
    num_results = 0
    is_matched = True
    for result in all_results:
      num_results += 1
      is_matched = is_matched and (result["Model"] in llms) and (
          result["Prompt"] in templates)  # check LLMs and templates
      is_matched = is_matched and (result["num_eval"] == num_eval) and (
          result["num_splits"] == num_splits)  # check num_eval and num_splits
      is_matched = is_matched and (result["num_runs"] == num_runs) and (
          result["split_id"] == split_id)  # check num_runs and split_id
    is_matched = is_matched and (
        num_results == (len(llms) * len(templates))
    )  # check the number of unique llms and templates
    return is_matched

  def compute_self_consistency(self,
                               dataset_id,
                               llm,
                               template,
                               split_id,
                               num_runs,
                               num_eval,
                               use_cache=True):
    """
    Compute self-consistency results.
    Args:
      - dataset_id: data task (e.g. summarize, hhrlhf-helpful).
      - llm: large language model judge name.
      - template: prompt template used for judging.
      - split_id: which split used to compute self-consistency results.
      - num_runs: number of repetition to compute self-consistency results.
      - num_eval: number of samples to be evaluated for each split.
      - use_cache: if using the cached metrics computation results.
    """
    judge_name = self.make_judge_name(llm, template)
    is_valid_judge_results, judge_results_files = self._check_judge_results(
        dataset_id, judge_name, split_id, num_runs)
    # Check if the judging results to compute self consistency exist
    assert is_valid_judge_results, f"Judging results to compute self-consistency do not exist!"
    # Get the self-consistency results
    full_cache_path = os.path.join(
        self.cache_dir, dataset_id, "metrics_results", judge_name,
        f"**.self_consist_results.{dataset_id}.{judge_name}.jsonl")
    if use_cache and self._check_cached_self_consistency_results(
        full_cache_path, split_id, num_eval, num_runs):
      # Read the cached self-consistency results for all the splits
      print(
          f">> Loading cached self-consistency results from {find_latest(full_cache_path)} ..."
      )
      self_consist_results = next(
          iter(jsonl_file_read(find_latest(full_cache_path))))
    else:
      # Get the self-consistency results
      print(">> Compute self-consistency results ...")
      self_consist_results = evaluate_self_consistency(judge_results_files,
                                                       llm, template)
      # Add fields for checking the cached results
      self_consist_results["split_id"] = split_id
      self_consist_results["num_runs"] = num_runs
      self_consist_results["num_eval"] = num_eval
      # Write it into the cached file
      output_path = os.path.join(self.cache_dir, dataset_id, "metrics_results",
                                 judge_name)
      new_dir(output_path)
      output_path = os.path.join(
          output_path,
          f"{date.today().strftime('%Y_%m_%d')}.self_consist_results.{dataset_id}.{judge_name}.jsonl"
      )
      jsonl_file_write(self_consist_results, output_path)
    print("Done.")
    return self_consist_results

  def compute_acc_bias(self,
                       judge_results,
                       dataset_id,
                       llm,
                       template,
                       num_splits,
                       num_eval,
                       use_cache=True):
    """
    Compute accuracy (both and random), position bias and length bias with no mitigation of flipping noise.
    The results contains computed metrics from all the splits.
    Args:
      - judge_results: dictionary of judging results from all the splits.
      - dataset_id: data task (e.g. summarize, hhrlhf-helpful).
      - llm: large language model judge name.
      - template: prompt template used for judging.
      - num_splits: number of splits to measure accuracy and bias.
      - num_eval: number of samples to be evaluated for each split.
      - use_cache: if using the cached metrics computation results.
    """
    # Check the correctness
    assert len(
        judge_results
    ) == num_splits, "please check how many splits in the judging results!"
    # Compute computation results aggregated for all the splits
    judge_name = self.make_judge_name(llm, template)
    full_cache_path = os.path.join(
        self.cache_dir, dataset_id, "metrics_results", judge_name,
        f"**.acc_bias_results.{dataset_id}.{judge_name}.jsonl")
    if use_cache and self._check_cached_acc_bias_results(
        full_cache_path, num_splits, num_eval):
      # Read the cached aggregated results for all the splits
      print(
          f">> Loading cached accuracy bias results from {find_latest(full_cache_path)}..."
      )
      acc_bias_results = next(
          iter(jsonl_file_read(find_latest(full_cache_path))))
    else:
      # Get the result for all the splits
      print(">> Computing accuracy bias results ...")
      results_all_splits = []
      for split_id in range(num_splits):
        results = analyze_separate_rewarding_accuracy_bias(
            judge_results[split_id])
        results["split_id"] = split_id
        results_all_splits.append(results)
      # Aggregate the results
      acc_bias_results = format_method_comparison_results(
          llm, template, results_all_splits)
      # Add the number of evaluted samples (for reusing the correct cached results)
      acc_bias_results["num_eval"] = num_eval
      # Write it into the cache directory
      output_path = os.path.join(self.cache_dir, dataset_id, "metrics_results",
                                 judge_name)
      new_dir(output_path)
      output_path = os.path.join(
          output_path,
          f"{date.today().strftime('%Y_%m_%d')}.acc_bias_results.{dataset_id}.{judge_name}.jsonl"
      )
      jsonl_file_write(acc_bias_results, output_path)
    print("Done.")

    return acc_bias_results

  def post_process_results(self,
                           acc_bias_results,
                           self_consist_results,
                           dataset_id,
                           llms,
                           templates,
                           num_eval,
                           num_splits,
                           num_runs,
                           split_id,
                           use_cache=True):
    """
    Post process results from all LLM judges.
    Args:
      - acc_bias_results: list of accuracy bias results (each result is contained in a dictionary).
      - self_consist_results: list of self consistency results (each result is contained in a dictionary).
      - llms: list of LLM names.
      - templates: list of prompt template names.
      - num_eval: number of evaluated samples for each split.
      - num_splits: number of splits for each LLM judge.
      - num_runs: number of repetition to compute the self-consistency results.
      - use_cache: if use cached results. Defaults to True.
    """
    # Check the cachced file
    if use_cache and self._check_post_processed_results(
        dataset_id, llms, templates, num_eval, num_splits, num_runs, split_id):
      full_cache_path = find_latest(
          os.path.join(self.cache_dir, dataset_id, "all_results", "**"))
      full_cache_path = os.path.join(full_cache_path,
                                     f"all_results.{dataset_id}.jsonl")
      print(
          f">> Loading cached post-processed results from {find_latest(full_cache_path)}..."
      )
      all_results = [result for result in jsonl_file_read(full_cache_path)]
      all_results = pd.DataFrame(all_results, dtype=str)  # dataframe
      # Remove added key fields from the dataframe (they are kept in the jsonl file)
      all_results = all_results.drop(
          ["num_eval", "num_splits", "num_runs", "split_id"], axis=1)
    else:
      all_results, out_path = aggregate_results(acc_bias_results,
                                                self_consist_results,
                                                dataset_id, llms, templates,
                                                self.cache_dir, num_eval,
                                                num_splits, num_runs,
                                                split_id)  # dataframe
      split_results(all_results, out_path)
      get_rank(all_results, out_path)
    print("Done.")

    return all_results
