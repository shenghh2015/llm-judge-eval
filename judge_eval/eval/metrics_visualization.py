"""
Visualize Accuracy (Both, Random), Position Bias and Length Bias.
"""

from judge_eval.utils.others import new_dir, find_latest
from judge_eval.utils.read_write import jsonl_file_read, jsonl_file_write
from judge_eval.utils.visualize import visualize_marginal, visualize_position_bias_accuracy, visualize_length_bias_accuracy
from datetime import date
import os


class Visualization:
  """
  Visualize Accuracy (Both, Random), Position Bias and Length Bias and their relationships.
  """

  def __init__(self, dataset_id, cache_dir):
    self.dataset_id = dataset_id  # data task (e.g. summarize, hhrlhf_helpful)
    self._cache_dir = cache_dir  # directory to store visualization results

  def _check_cached_visual_results(self, llms, templates, num_eval, num_splits,
                                   num_runs, split_id):
    latest = find_latest(
        os.path.join(self._cache_dir, self.dataset_id, "visualize_results",
                     "**"))
    if latest == '': return False
    # Level 0: Check visualization files
    is_exist = True
    is_exist = is_exist and os.path.exists(
        os.path.join(latest,
                     f"viz_results.{self.dataset_id}.accuracy_both.png"))
    is_exist = is_exist and os.path.exists(
        os.path.join(latest,
                     f"viz_results.{self.dataset_id}.accuracy_random.png"))
    is_exist = is_exist and os.path.exists(
        os.path.join(latest,
                     f"viz_results.{self.dataset_id}.position_bias.png"))
    is_exist = is_exist and os.path.exists(
        os.path.join(latest, f"viz_results.{self.dataset_id}.length_bias.png"))
    is_exist = is_exist and os.path.exists(
        os.path.join(
            latest,
            f"viz_results.{self.dataset_id}.position_bias_accuracy.png"))
    is_exist = is_exist and os.path.exists(
        os.path.join(
            latest, f"viz_results.{self.dataset_id}.length_bias_accuracy.png"))
    is_exist = is_exist and os.path.exists(
        os.path.join(latest, "meta_info.jsonl"))
    if not is_exist: return False
    # Level 1: Check the meta info
    is_match = True
    meta_info = next(
        iter(jsonl_file_read(os.path.join(latest, f"meta_info.jsonl"))))
    is_match = is_match and sorted(meta_info["models"]) == sorted(llms)
    is_match = is_match and sorted(meta_info["templates"]) == sorted(templates)
    is_match = is_match and meta_info["num_eval"] == num_eval
    is_match = is_match and meta_info["num_splits"] == num_splits
    is_match = is_match and meta_info["num_runs"] == num_runs
    is_match = is_match and meta_info["split_id"] == split_id
    return is_match

  def draw_accuracy(self, df_results, out_folder):
    visualize_marginal(df_results, out_folder, metric="Accuracy (Both)")
    visualize_marginal(df_results, out_folder, metric="Accuracy (Random)")

  def draw_position_bias(self, df_results, out_folder):
    visualize_marginal(df_results, out_folder, metric="Position Bias")

  def draw_length_bias(self, df_results, out_folder):
    visualize_marginal(df_results,
                       out_folder,
                       metric="Length Bias (All Pairs)")

  def draw_PB_Acc(self, df_results, out_folder):
    visualize_position_bias_accuracy(df_results, out_folder)

  def draw_LB_Acc(self, df_results, out_folder):
    visualize_length_bias_accuracy(df_results, out_folder)

  def write_key_fields(self, llms, templates, num_eval, num_splits, num_runs,
                       split_id, out_folder):
    '''Write key fields for checking the cached visualization results.'''
    meta_info = {
        "models": llms,
        "templates": templates,
        "num_eval": num_eval,
        "num_splits": num_splits,
        "num_runs": num_runs,
        "split_id": split_id
    }

    jsonl_file_write(meta_info, os.path.join(out_folder, "meta_info.jsonl"))

  def visualize_metrics(self,
                        df_results,
                        llms,
                        templates,
                        num_eval,
                        num_splits,
                        num_runs,
                        split_id,
                        use_cache=True):
    """
    Visualize Accuracy, Position Bias, Length Bias and their relationships.
    Args:
      - df_results: datframe which contains all the results of all the tested LLM judges.
      - from_scratch: if re-draw all the visualization results. Defaults to False.
      - llms: list of tested LLM names.
      - templates: list of tested prompt templates.
      - num_eval: number of samples to be evaluated for each split.
      - num_splits: number of splits to measure accuracy and bias.
      - num_runs: number of repetition to compute self-consistency results.
      - split_id: index of the split to compute self-consistency results.
      - use_cache: if the cached visualization results have existed.
    """

    if use_cache and self._check_cached_visual_results(
        llms, templates, num_eval, num_splits, num_runs, split_id):
      cached_results = find_latest(
          os.path.join(self._cache_dir, self.dataset_id, "visualize_results",
                       "**"))
      print(
          f"Please check the cached visualization results at the folder {cached_results}"
      )
    else:
      # Sort the results by llm-model and template names
      df_results = df_results.sort_values(by=['Model', 'Prompt'],
                                          ascending=[True, True])
      out_folder = os.path.join(self._cache_dir, self.dataset_id,
                                "visualize_results",
                                date.today().strftime('%Y-%m-%d'))
      new_dir(out_folder)

      # Accuracy (Both) and Accuracy (Random)
      self.draw_accuracy(df_results, out_folder)

      # Denoised Position Bias
      self.draw_position_bias(df_results, out_folder)

      # Denoised Length Bias
      self.draw_length_bias(df_results, out_folder)

      # Position Bias vs. Accuracy (Both)
      self.draw_PB_Acc(df_results, out_folder)

      # Length Bias vs. Accuracy (Both)
      self.draw_LB_Acc(df_results, out_folder)

      # Write meta information
      self.write_key_fields(llms, templates, num_eval, num_splits, num_runs,
                            split_id, out_folder)
