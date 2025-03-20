from judge_eval.utils.others import new_dir
from judge_eval.utils.read_write import jsonl_file_write
from scipy.stats import ttest_rel
from datetime import date
import regex
import numpy as np
import pandas as pd
import os


def get_denoised_position_bias(acc_bias_result, self_consist_result):
  """
  Get the position bias P(A)-P(B) after removing the impact of flipping noise.
  """
  # Get the flipping noise (i.e. 1-p_consist)
  noise_A, noise_B = 1 - self_consist_result[
      "p_consist_A"], 1 - self_consist_result["p_consist_B"]
  # Compute the denoised position bias
  acc_A_noisy = np.array([
      float(acc_A) for acc_A in
      acc_bias_result["Accuracy (A)"].values[0].split("|")[0].split(",")
  ])
  acc_B_noisy = np.array([
      float(acc_B) for acc_B in
      acc_bias_result["Accuracy (B)"].values[0].split("|")[0].split(",")
  ])
  acc_A_denoise = (acc_A_noisy - noise_A) / (1 - 2 * noise_A + 1e-16)
  acc_B_denoise = (acc_B_noisy - noise_B) / (1 - 2 * noise_B + 1e-16)
  position_bias = acc_A_denoise - acc_B_denoise
  # Add the denoised accuracy back into the dataframe
  acc_bias_result["Position Bias"] = ", ".join(
      map("{:.3f}".format, position_bias)
  ) + " | " + f"{np.mean(position_bias):.3f} ({np.std(position_bias):.3f})"

  return acc_bias_result


def get_denoised_length_bias(acc_bias_result, self_consist_result):
  """
  Get the length bias Acc(Human choose long)-Acc(Human choose short) 
  after removing the impact of flipping noise.
  """
  # Get the flipping noise (i.e. 1-p_consist) for long and others
  noise_long, noise_short = 1 - self_consist_result[
      "p_consist_long"], 1 - self_consist_result["p_consist_short"]
  # Compute the denoised position bias
  acc_long_noisy = np.array([
      float(acc_long)
      for acc_long in acc_bias_result["Accuracy_long (Both)"].values[0].split(
          "|")[0].split(",")
  ])
  acc_short_noisy = np.array([
      float(acc_short)
      for acc_short in acc_bias_result["Accuracy_others (Both)"].values[0].
      split("|")[0].split(",")
  ])
  acc_long_denoise = (acc_long_noisy - noise_long) / (1 - 2 * noise_long +
                                                      1e-16)
  acc_short_denoise = (acc_short_noisy - noise_short) / (1 - 2 * noise_short +
                                                         1e-16)
  length_bias_split = acc_long_denoise - acc_short_denoise  # No absolute value
  length_bias_all_pairs = acc_long_denoise[..., None] - acc_short_denoise[
      None, ...]  # No absolute value
  # Add the denoised accuracy back into the dataframe
  acc_bias_result["Length Bias (Splits)"] = ", ".join(
      map("{:.3f}".format, length_bias_split)
  ) + " | " + f"{np.mean(length_bias_split):.3f} ({np.std(length_bias_split):.3f})"
  acc_bias_result[
      "Length Bias (All Pairs)"] = f"{np.mean(length_bias_all_pairs.ravel()):.3f} ({np.std(length_bias_all_pairs.ravel()):.3f})"
  acc_bias_result["Accuracy_long_denoise (Both)"] = ", ".join(
      map("{:.3f}".format, acc_long_denoise)
  ) + " | " + f"{np.mean(acc_long_denoise):.3f} ({np.std(acc_long_denoise):.3f})"
  acc_bias_result["Accuracy_others_denoise (Both)"] = ", ".join(
      map("{:.3f}".format, acc_short_denoise)
  ) + " | " + f"{np.mean(acc_short_denoise):.3f} ({np.std(acc_short_denoise):.3f})"

  return acc_bias_result


def aggregate_results(acc_bias_results, self_consist_results, dataset_id, llms,
                      templates, out_path, num_eval, num_splits, num_runs,
                      split_id):
  """
  Stack the results from different methods (i.e. LLM judges).
  Args:
    - acc_bias_results: list of accuracy and bias results (each result is from one method).
    - self_consist_results: list of self-consistency results (each result is from one method).
    - dataset_id: data task (e.g. summarize, hhrlhf-helpful).
    - llms: list of tested LLM names.
    - templates: list of tested prompt template names.
    - out_path: output directory which stores the aggregated results.
    - num_eval: number of evaluated samples for each split.
    - num_splits: number of splits for each LLM judge.
    - num_runs: number of repetition to compute the self-consistency results.
    - split_id: index of the split used to compute the self-consistency results.
  """
  # Check inputs
  num_llms = len(llms)
  num_temps = len(templates)
  assert len(acc_bias_results) == (
      num_llms *
      num_temps), "results of accuracy and bias have wrong number of files!"
  assert len(self_consist_results) == (
      num_llms *
      num_temps), "results of self-consistency have wrong number of files!"
  # Aggregate results
  all_results = []
  for ldx, model in enumerate(llms):
    for tdx, template in enumerate(templates):
      ab_result, sc_result = acc_bias_results[ldx * len(templates) +
                                              tdx], self_consist_results[
                                                  ldx * len(templates) + tdx]
      assert (ab_result["Model"] == model) and (
          sc_result["Model"] == model
      ), "When aggregating results, LLM name orders are different in provided llm list and result list!"
      assert (ab_result["Prompt"] == template) and (
          sc_result["Prompt"] == template
      ), "When aggregating results, template name orders are different in provided llm list and result list!"
      ab_result = pd.DataFrame([ab_result], dtype=str)
      df_result = get_denoised_position_bias(ab_result, sc_result)
      df_result = get_denoised_length_bias(ab_result, sc_result)
      # Add key fields for checking the cached file
      ab_result["num_eval"], ab_result["num_splits"] = num_eval, num_splits
      ab_result["num_runs"], ab_result["split_id"] = num_runs, split_id
      all_results.append(df_result)
  df_all_results = pd.concat(all_results)
  agg_results = df_all_results.to_dict("records")  # a list of dictionaries
  # Write the aggregated result into .jsonl file
  out_path = os.path.join(out_path, dataset_id, "all_results",
                          date.today().strftime('%Y-%m-%d'))
  new_dir(out_path)
  out_filename = f"all_results.{dataset_id}.jsonl"
  out_path = os.path.join(out_path, out_filename)
  print(f"Write the aggregated result into {out_path}")
  jsonl_file_write(agg_results, out_path)
  # Remove added key fields from the dataframe (they are kept in the jsonl file)
  df_all_results = df_all_results.drop(
      ["num_eval", "num_splits", "num_runs", "split_id"], axis=1)

  return df_all_results, out_path


def get_result_per_column(series, split_index):
  '''Get the result of each splits.'''
  series = series.str.split("|")
  series = series.apply(lambda x: x[0])
  series = series.str.split(",")
  series = series.apply(lambda x: x[split_index])
  series = series.str.strip()

  return series


def get_result_overall(series):
  '''Get the overall results. It can handle cases with or without std.'''
  series = series.str.split("|")
  series = series.apply(lambda x: x[-1])
  series = series.str.split("(")
  series = series.apply(lambda x: x[0])
  series = series.str.strip()

  return series


def get_results_all_splits(series):
  '''Get the results of all the splits.'''
  series = series.str.split("|")
  series = series.apply(lambda x: x[0])
  series = series.str.split(",")
  series = series.apply(lambda x: [float(num.strip()) for num in x])

  return series


def check_value_depend_on_all_splits(series):
  '''Check if all values in a column depends on all the splits.'''
  return len(series.iloc[0].split('|')) > 1


def get_number_of_splits(df):
  '''Get how many splits are aggregated.'''
  return len(df.iloc[0]["Valid Count"].split("|")[0].split(","))


def split_results(df_results, out_path):
  """
  Split the dataframe containing all the splits and all the methods into 
  dataframes which only contain the results for each split.
  Args:
    - df_results: dataframe containing the results of all the splits and the methods.
    - out_path: output path of the aggregated result, 
                used to store the split results under the same directory.
  """
  # Select columns that meet the condition
  columns_to_modify = [
      col for col in df_results.columns
      if check_value_depend_on_all_splits(df_results[col])
  ]
  # Select columnts that will not keep in the final result for each split
  columns_to_remove = [
      col for col in df_results.columns
      if (col not in columns_to_modify) and (col not in ["Model", "Prompt"])
  ]
  # Get the number of splits
  num_splits = get_number_of_splits(df_results)
  # Split the results
  for split_index in range(num_splits):
    df_split = df_results.copy()
    df_split[columns_to_modify] = df_split[columns_to_modify].apply(
        get_result_per_column, args=(split_index, ), axis=1)
    df_split = df_split.drop(columns=columns_to_remove)
    out_path_split = os.path.join(
        *out_path.split("/")[:-1],
        out_path.split("/")[-1].replace("all_results",
                                        f"split{split_index}_results"))
    result_split = df_split.to_dict("records")
    jsonl_file_write(result_split, out_path_split)
    print(f"Write the split {split_index} result into {out_path_split}")


def get_rank_split(df_results, metric, split_index):
  '''Rank each method (model + prompt) based on the metric for the split.'''
  if not check_value_depend_on_all_splits(df_results[metric]):
    return "N/A"
  df_split = df_results[["Model", "Prompt", metric]].copy()
  df_split[metric] = get_result_per_column(df_split[metric],
                                           split_index).astype(np.float32)
  df_split["Method"] = df_split.apply(lambda x: f'{x["Model"]}:{x["Prompt"]}',
                                      axis=1)
  if regex.match(r"Valid Count|(?:Minimum\s|Maximum\s)?Accuracy", metric):
    is_ascending = False
    compare_symbol = '>'
  else:
    is_ascending = True
    compare_symbol = '<'
  df_split = df_split.sort_values(metric, ascending=is_ascending)

  return f' {compare_symbol} '.join(df_split["Method"].values)


def get_rank_overall(df_results, metric):
  '''Rank each method (model + prompt) based on the metric.'''
  # Get the necessary columns
  if metric == "All Pairs Difference (Both)":
    df_results = df_results[[
        "Model", "Prompt", metric, "Accuracy_long (Both)",
        "Accuracy_others (Both)"
    ]].copy()
    df_results["Accuracy_long (Both)"] = get_results_all_splits(
        df_results["Accuracy_long (Both)"])
    df_results["Accuracy_others (Both)"] = get_results_all_splits(
        df_results["Accuracy_others (Both)"])

  elif metric == "All Pairs Difference (Random)":
    df_results = df_results[[
        "Model", "Prompt", metric, "Accuracy_long (Random)",
        "Accuracy_others (Random)"
    ]].copy()
    df_results["Accuracy_long (Random)"] = get_results_all_splits(
        df_results["Accuracy_long (Random)"])
    df_results["Accuracy_others (Random)"] = get_results_all_splits(
        df_results["Accuracy_others (Random)"])

  elif metric == "Length Bias (All Pairs)":
    df_results = df_results[[
        "Model", "Prompt", metric, "Accuracy_long_denoise (Both)",
        "Accuracy_others_denoise (Both)"
    ]].copy()
    df_results["Accuracy_long_denoise (Both)"] = get_results_all_splits(
        df_results["Accuracy_long_denoise (Both)"])
    df_results["Accuracy_others_denoise (Both)"] = get_results_all_splits(
        df_results["Accuracy_others_denoise (Both)"])

  else:
    df_results = df_results[["Model", "Prompt", metric]].copy()
    df_results["All Split Results"] = get_results_all_splits(
        df_results[metric])

  # Get the rank
  df_results[metric] = get_result_overall(df_results[metric]).astype(
      np.float32)
  df_results["Method"] = df_results.apply(
      lambda x: f'{x["Model"]}:{x["Prompt"]}', axis=1)
  if regex.match(r"Valid Count|(?:Minimum\s|Maximum\s)?Accuracy", metric):
    is_ascending = False
    compare_symbol = '>'
  else:
    is_ascending = True
    compare_symbol = '<'
  df_results = df_results.sort_values(metric, ascending=is_ascending)

  def get_diff_all_pairs(row, col1, col2):
    '''Compute the p-value between each pair'''
    return (np.array(row[col1])[..., None] -
            np.array(row[col2])[None, ...]).ravel()

  def get_pval(series):
    p_values = []
    for i in range(len(series) - 1):
      row1 = series.iloc[i]
      row2 = series.iloc[i + 1]
      p_val = ttest_rel(row1, row2).pvalue
      p_values.append(f"{p_val:.4f}")
    return p_values

  if metric == "All Pairs Difference (Both)":
    df_results["All Split Results"] = df_results.apply(
        lambda x: get_diff_all_pairs(x, "Accuracy_long (Both)",
                                     "Accuracy_others (Both)"),
        axis=1)

  if metric == "All Pairs Difference (Random)":
    df_results["All Split Results"] = df_results.apply(
        lambda x: get_diff_all_pairs(x, "Accuracy_long (Random)",
                                     "Accuracy_others (Random)"),
        axis=1)

  if metric == "Length Bias (All Pairs)":
    df_results["All Split Results"] = df_results.apply(
        lambda x: get_diff_all_pairs(x, "Accuracy_long_denoise (Both)",
                                     "Accuracy_others_denoise (Both)"),
        axis=1)

  if len(df_results["All Split Results"].iloc[0]) == 1:
    pval_format = "N/A"
  else:
    p_values = get_pval(df_results["All Split Results"])
    pval_format = ' , '.join(p_values)

  return f' {compare_symbol} '.join(
      df_results["Method"].values) + ' | ' + pval_format


def get_rank(df_results, out_path):
  """
  Rank different methods based on different metrics. 
  The ranks are on each split and the overall average result.
  Args:
    - df_results: dataframe containing the results of all the splits and the methods.
    - out_path: output path of the aggregated result, 
                used to store the split results under the same directory.
  """
  results = []
  num_splits = get_number_of_splits(df_results)
  for metric in df_results.columns:
    if regex.match(r"P-value|Model|Prompt$", metric) is not None:
      continue
    rank_metric = {"Metric": metric}
    # Get the overall rank result (depends on all the splits)
    rank_metric[f"Overall"] = get_rank_overall(df_results, metric)
    # Get the rank result for each split
    for split_idx in range(num_splits):
      rank_metric[f"Split {split_idx}"] = get_rank_split(
          df_results, metric, split_idx)
    results.append(rank_metric)
  # Write the rank results into .jsonl file
  out_path = os.path.join(
      *out_path.split("/")[:-1],
      out_path.split("/")[-1].replace("all_results", "rank_results"))
  jsonl_file_write(results, out_path)
  print(f'Write the rank results into {out_path}')
