from judge_eval.configs.column_name_map import NAME_MAP
import random
import numpy as np
from scipy.stats import ttest_ind


def get_accuracy_combine(data, random_num):
  """
  Get the valid count and accuracy for both and random positions.
  """
  valid_count, acc, random_acc = 0, 0, 0
  if 'chosen_score_A' in data:
    if data['chosen_score_A'] !=-1 and data['reject_score_A'] != -1 \
       and data['chosen_score_B'] !=-1 and data['reject_score_B'] != -1:
      acc = (data['chosen_score'] > data['reject_score']) * 1
      valid_count = 1
      # random position
      if random_num < 0.5:
        random_acc = (data['chosen_score_A'] > data['reject_score_A']) * 1
      else:
        random_acc = (data['chosen_score_B'] > data['reject_score_B']) * 1
  else:
    if data['chosen_score'] != -1 and data['reject_score'] != -1:
      acc = (data['chosen_score'] > data['reject_score']) * 1
      valid_count = 1

  return valid_count, acc, random_acc


def get_position_bias(data):
  """
  Get metrics for position bias.
  """
  inconsistent, chosen_A, chosen_B = 0, 0, 0
  if 'chosen_score_A' in data:
    if data['chosen_score_A'] !=-1 and data['reject_score_A'] != -1 \
      and data['chosen_score_B'] !=-1 and data['reject_score_B'] != -1:
      inconsistent = (data['chosen_score'] == data['reject_score']) * 1
      chosen_A = (data['chosen_score'] > data['reject_score']) * 1
      chosen_B = (data['chosen_score'] < data['reject_score']) * 1

  return inconsistent, chosen_A, chosen_B


def get_length_bias(data, random_num):
  """
  Get metrics for length bias.
  """
  acc_long, acc_not_long = 0, 0
  random_acc_long, random_acc_not_long = 0, 0
  num_long, num_not_long = 0, 0
  if 'chosen_score_A' in data:
    if data['chosen_score_A'] !=-1 and data['reject_score_A'] != -1 \
      and data['chosen_score_B'] !=-1 and data['reject_score_B'] != -1:
      acc_long = ((len(data['chosen']) > len(data['reject'])) and
                  (data['chosen_score'] > data['reject_score'])) * 1
      acc_not_long = ((len(data['chosen']) <= len(data['reject'])) and
                      (data['chosen_score'] > data['reject_score'])) * 1
      # random position
      if random_num < 0.5:
        random_acc_long = (
            (len(data['chosen']) > len(data['reject'])) and
            (data['chosen_score_A'] > data['reject_score_A'])) * 1
        random_acc_not_long = (
            (len(data['chosen']) <= len(data['reject'])) and
            (data['chosen_score_A'] > data['reject_score_A'])) * 1
      else:
        random_acc_long = (
            (len(data['chosen']) > len(data['reject'])) and
            (data['chosen_score_B'] > data['reject_score_B'])) * 1
        random_acc_not_long = (
            (len(data['chosen']) <= len(data['reject'])) and
            (data['chosen_score_B'] > data['reject_score_B'])) * 1
      num_long = (len(data["chosen"]) > len(data["reject"])) * 1
      num_not_long = (len(data["chosen"]) <= len(data["reject"])) * 1
  else:
    if data['chosen_score'] != -1 and data['reject_score'] != -1:
      acc_long = ((len(data['chosen']) > len(data['reject'])) and
                  (data['chosen_score'] > data['reject_score'])) * 1
      acc_not_long = ((len(data['chosen']) <= len(data['reject'])) and
                      (data['chosen_score'] > data['reject_score'])) * 1
      num_long = (len(data["chosen"]) > len(data["reject"])) * 1
      num_not_long = (len(data["chosen"]) <= len(data["reject"])) * 1

  return acc_long, acc_not_long, random_acc_long, random_acc_not_long, num_long, num_not_long


def get_prompt_response_length(data, random_num):
  """
  Get the length of prompt and response.
  """
  len_prompt, len_response, len_prompt_random, len_response_random = 0, 0, 0, 0
  if 'chosen_score_A' in data:
    if data['chosen_score_A'] !=-1 and data['reject_score_A'] != -1 \
      and data['chosen_score_B'] !=-1 and data['reject_score_B'] != -1:
      len_prompt = data['length_prompt_A'] + data['length_prompt_B']
      len_response = data['length_response_A'] + data['length_response_B']
      # random position
      if random_num < 0.5:
        len_prompt_random = data['length_prompt_A']
        len_response_random = data['length_response_A']
      else:
        len_prompt_random = data['length_prompt_B']
        len_response_random = data['length_response_B']
  else:
    if data['chosen_score'] != -1 and data['reject_score'] != -1:
      len_prompt = data['length_prompt']
      len_response = data['length_response']
  return len_prompt, len_response, len_prompt_random, len_response_random


def get_run_time(data, random_num):
  """
  Get the run time.
  """
  run_time, run_time_random = 0, 0
  if 'chosen_score_A' in data:
    if data['chosen_score_A'] !=-1 and data['reject_score_A'] != -1 \
      and data['chosen_score_B'] !=-1 and data['reject_score_B'] != -1:
      run_time = data['run_time_A'] + data['run_time_B']
      # random position
      if random_num < 0.5:
        run_time_random = data['run_time_A']
      else:
        run_time_random = data['run_time_B']
  else:
    if data['chosen_score'] != -1 and data['reject_score'] != -1:
      run_time = data['run_time']
  return run_time, run_time_random


def get_accuracy_separate(data):
  """
  Get accuracy for each position (chosen-reject and reject-chosen).
  """
  valid_countA, valid_countB, acc_A, acc_B = 0, 0, 0, 0
  chosen_score_A = data['chosen_score_A']
  chosen_score_B = data['chosen_score_B']
  reject_score_A = data['reject_score_A']
  reject_score_B = data['reject_score_B']
  if chosen_score_A != -1 and reject_score_A != -1:
    acc_A = (chosen_score_A > reject_score_A) * 1
    valid_countA = 1
  if chosen_score_B != -1 and reject_score_B != -1:
    acc_B = (chosen_score_B > reject_score_B) * 1
    valid_countB = 1
  return valid_countA, valid_countB, acc_A, acc_B


def analyze_separate_rewarding_accuracy_bias(datalist):
  """
  Analyze the results and measure the accuracy and bias 
  of the method (model + template) for each split.
  """
  # Set random seed to compute Acc (Random)
  random.seed(5)

  # Initialize computation metrics
  # valid count
  valid_count = 0
  # accuracy
  acc = 0
  random_acc = 0
  # position bias
  inconsistent = 0
  chosen_A = 0
  chosen_B = 0
  # length bias
  acc_long = 0
  acc_not_long = 0
  random_acc_long = 0
  random_acc_not_long = 0
  num_long = 0
  num_not_long = 0
  # prompt and response length
  len_prompt = 0
  len_response = 0
  len_prompt_random = 0
  len_response_random = 0
  # run time
  run_time = 0.0
  run_time_random = 0.0

  # Compute metrics
  for _data in datalist:
    random_num = random.random()

    result = get_accuracy_combine(_data, random_num)
    valid_count += result[0]
    acc += result[1]
    random_acc += result[2]

    result = get_position_bias(_data)
    inconsistent += result[0]
    chosen_A += result[1]
    chosen_B += result[2]

    result = get_length_bias(_data, random_num)
    acc_long += result[0]
    acc_not_long += result[1]
    random_acc_long += result[2]
    random_acc_not_long += result[3]
    num_long += result[4]
    num_not_long += result[5]

    result = get_prompt_response_length(_data, random_num)
    len_prompt += result[0]
    len_response += result[1]
    len_prompt_random += result[2]
    len_response_random += result[3]

    result = get_run_time(_data, random_num)
    run_time += result[0]
    run_time_random += result[1]

  acc = round(acc / (valid_count + 1e-16), 3)
  print("*-" * 50)
  print(f"> Total count: {len(datalist)}, valid count: {valid_count}")
  print(f'> Accuracy (both positions): {acc}')

  if not ('chosen_score_A' in datalist[0] and 'chosen_score_B' in datalist[0]):
    avg_len_prompt = round(len_prompt / (valid_count + 1e-16), 3)
    avg_len_response = round(len_response / (valid_count + 1e-16), 3)
    avg_run_time = round(run_time / (valid_count + 1e-16), 3)
    print(f'> Prompt length per valid count: {avg_len_prompt:.3f}')
    print(f'> Response length per valid count: {avg_len_response:.3f}')
    print(f'> Run time per valid count: {avg_run_time:.3f}')
    return {
        "valid_count": valid_count,
        "acc": acc,
        "len_prompt": avg_len_prompt,
        "len_response": avg_len_response,
        "run_time": avg_run_time
    }

  # Analysis for each position (chosen-reject and reject-chosen)
  valid_countA = 0
  valid_countB = 0
  acc_A = 0
  acc_B = 0
  for _data in datalist:
    result = get_accuracy_separate(_data)
    valid_countA += result[0]
    valid_countB += result[1]
    acc_A += result[2]
    acc_B += result[3]
  acc_A = round(acc_A / (valid_countA + 1e-16), 3)
  acc_B = round(acc_B / (valid_countB + 1e-16), 3)
  acc_AB = round((acc_A + acc_B) / 2, 3)
  max_acc_AB = max(acc_A, acc_B)
  min_acc_AB = min(acc_A, acc_B)
  random_acc = round(random_acc / (valid_count + 1e-16), 3)
  print(f'> Accuracy (random position): {random_acc}')
  print(f'> Accuracy (chosen-reject position): {acc_A}')
  print(f'> Accuracy (reject-chosen position): {acc_B}')
  print(f'> Accuracy (average of both positions): {acc_AB}')
  print(f'> Accuracy (max of both positions): {max_acc_AB}')
  print(f'> Accuracy (min of both positions): {min_acc_AB}')
  print()

  # Position bias, defined by inconsistent rate
  # This version does not mitigate the flipping noise
  inconsistent_rate = round(inconsistent / (valid_count + 1e-16), 3)
  print("*-" * 50)
  print(
      f'> Valid count: {valid_count}, chose A count: {chosen_A}, chose B count: {chosen_B}, inconsistent count: {inconsistent}'
  )
  print(f'> Inconsistent Rate (both positions): {inconsistent_rate}')
  print()

  # Length bias
  # This version does not mitigate the flipping noise.
  acc_long = round(acc_long / (num_long + 1e-16), 3)
  acc_not_long = round(acc_not_long / (num_not_long + 1e-16), 3)
  random_acc_long = round(random_acc_long / (num_long + 1e-16), 3)
  random_acc_not_long = round(random_acc_not_long / (num_not_long + 1e-16), 3)
  print("*-" * 50)
  print(f'> Count: long = {num_long} | others = {num_not_long}')
  print(
      f'> Accuracy (both positions): long = {acc_long} | others = {acc_not_long}'
  )
  print(
      f'> Accuracy (random position): long = {random_acc_long} | others = {random_acc_not_long}'
  )
  print()

  # Prompt and response length
  print("*-" * 50)
  avg_len_prompt = round(len_prompt / (valid_count + 1e-16), 3)
  avg_len_response = round(len_response / (valid_count + 1e-16), 3)
  avg_len_prompt_random = round(len_prompt_random / (valid_count + 1e-16), 3)
  avg_len_response_random = round(len_response_random / (valid_count + 1e-16),
                                  3)
  print(
      f'> Prompt length per valid count (both positions): {avg_len_prompt:.3f}'
  )
  print(
      f'> Response length per valid count (both positions): {avg_len_response:.3f}'
  )
  print(
      f'> Prompt length per valid count (random positions): {avg_len_prompt_random:.3f}'
  )
  print(
      f'> Response length per valid count (random positions): {avg_len_response_random:.3f}'
  )
  print()

  # Running time
  avg_run_time = round(run_time / (valid_count + 1e-16), 3)
  avg_run_time_random = round(run_time_random / (valid_count + 1e-16), 3)
  print(f'> Run time per valid count (both positions): {avg_run_time:.3f}')
  print(
      f'> Run time per valid count (random positions): {avg_run_time_random:.3f}'
  )

  return {
      "valid_count": valid_count,
      "acc": acc,
      "random_acc": random_acc,
      "valid_count_A": valid_countA,
      "valid_count_B": valid_countB,
      "acc_A": acc_A,
      "acc_B": acc_B,
      "acc_AB": acc_AB,
      "min_acc_AB": min_acc_AB,
      "max_acc_AB": max_acc_AB,
      "inconsistent_rate": inconsistent_rate,
      "acc_long": acc_long,
      "acc_not_long": acc_not_long,
      "random_acc_long": random_acc_long,
      "random_acc_not_long": random_acc_not_long,
      "len_prompt": avg_len_prompt,
      "len_response": avg_len_response,
      "len_prompt_random": avg_len_prompt_random,
      "len_response_random": avg_len_response_random,
      "run_time": avg_run_time,
      "run_time_random": avg_run_time_random
  }


def format_method_comparison_results(llm, template, performs_all_splits):
  """
  Format the results for the final method comparison.
  """
  fancy_results = {"Model": llm, "Prompt": template}

  output_column_name = ["valid_count", "acc"]
  if "acc_A" in performs_all_splits[0].keys():
    output_column_name += ["random_acc", "acc_AB", "min_acc_AB", "max_acc_AB", "acc_A", "acc_B", "valid_count_A", "valid_count_B", \
                           "inconsistent_rate", \
                           "acc_long", "acc_not_long", "random_acc_long", "random_acc_not_long"]

  print()
  for name in output_column_name:
    perform_name = [perform[name] for perform in performs_all_splits]
    print(name.upper())
    print('> split_results:', ', '.join(map("{:.3f}".format, perform_name)))
    print(f"> mean: {np.mean(perform_name):.3f}")
    if len(perform_name) > 1:
      print(f"> std: {np.std(perform_name):.3f}")
    print("*-" * 50)
    print()
    str_format = ', '.join([f"{num:.3f}" for num in perform_name])
    str_format += f' | {np.mean(perform_name):.3f} ({np.std(perform_name):.3f})'
    fancy_results[NAME_MAP[name]] = str_format

  # evaluate length bias
  if ("acc_long" in output_column_name) and ("acc_not_long"
                                             in output_column_name):
    acc_long_list = np.array(
        [perform["acc_long"] for perform in performs_all_splits])
    acc_not_long_list = np.array(
        [perform["acc_not_long"] for perform in performs_all_splits])
    acc_mean_diff = np.mean(np.abs(acc_long_list - acc_not_long_list))
    acc_diff_mean = np.abs(np.mean(acc_long_list) - np.mean(acc_not_long_list))
    acc_diff_all_pairs = np.abs(acc_long_list[..., None] -
                                acc_not_long_list[None, ...])
    print(
        f"> mean of absolute difference (long vs. others) of accuracy (both positions): {acc_mean_diff:.3f}"
    )
    print(
        f"> absolute difference of mean (long vs. others) of accuracy (both positions): {acc_diff_mean:.3f}"
    )
    print(
        f"> all pairs absolute difference (long vs. others) of accuracy (both positions): mean(std) = {np.mean(acc_diff_all_pairs):.3f}({np.std(acc_diff_all_pairs):.3f})"
    )
    if len(acc_long_list) > 1:
      _, pval = ttest_ind(acc_long_list, acc_not_long_list)
      print(
          f"> p-value (long vs. others) of accuracy (both positions): {pval:.4f}"
      )
    print("*-" * 50)
    print()
    acc_abs_diff = np.abs(acc_long_list - acc_not_long_list)
    acc_diff_format = ', '.join([f"{num:.3f}" for num in acc_abs_diff])
    acc_diff_format += f' | {np.mean(acc_abs_diff):.3f} ({np.std(acc_abs_diff):.3f})'
    fancy_results["Mean Difference Accuracy (Both)"] = acc_diff_format
    fancy_results["Difference Mean Accuracy (Both)"] = f"{acc_diff_mean:.3f}"
    fancy_results[
        "All Pairs Difference (Both)"] = f"{np.mean(acc_diff_all_pairs):.3f} ({np.std(acc_diff_all_pairs):.3f})"
    if len(acc_long_list) > 1:
      fancy_results[
          "P-value (Both)"] = f"{pval:.4f}" if pval >= 0.0001 else "< 0.0001"

  if ("random_acc_long" in output_column_name) and ("random_acc_not_long"
                                                    in output_column_name):
    random_acc_long_list = np.array(
        [perform["random_acc_long"] for perform in performs_all_splits])
    random_acc_not_long_list = np.array(
        [perform["random_acc_not_long"] for perform in performs_all_splits])
    random_acc_mean_diff = np.mean(
        np.abs(random_acc_long_list - random_acc_not_long_list))
    random_acc_diff_mean = np.abs(
        np.mean(random_acc_long_list) - np.mean(random_acc_not_long_list))
    random_acc_diff_all_pairs = np.abs(random_acc_long_list[..., None] -
                                       random_acc_not_long_list[None, ...])
    print(
        f"> mean of absolute difference (long vs. others) of accuracy (random positions): {random_acc_mean_diff:.3f}"
    )
    print(
        f"> absolute difference of mean (long vs. others) of accuracy (random positions): {random_acc_diff_mean:.3f}"
    )
    print(
        f"> all pairs absolute difference (long vs. others) of accuracy (random positions): mean(std) = {np.mean(random_acc_diff_all_pairs):.3f}({np.std(random_acc_diff_all_pairs):.3f})"
    )
    if len(random_acc_long_list) > 1:
      _, pval = ttest_ind(random_acc_long_list, random_acc_not_long_list)
      print(
          f"> p-value (long vs. others) of accuracy (random positions): {pval:.4f}"
      )
    print("*-" * 50)
    print()
    random_acc_abs_diff = np.abs(random_acc_long_list -
                                 random_acc_not_long_list)
    random_acc_diff_format = ', '.join(
        [f"{num:.3f}" for num in random_acc_abs_diff])
    random_acc_diff_format += f' | {np.mean(random_acc_abs_diff):.3f} ({np.std(random_acc_abs_diff):.3f})'
    fancy_results["Mean Difference Accuracy (Random)"] = random_acc_diff_format
    fancy_results[
        "Difference Mean Accuracy (Random)"] = f"{random_acc_diff_mean:.3f}"
    fancy_results[
        "All Pairs Difference (Random)"] = f"{np.mean(random_acc_diff_all_pairs):.3f} ({np.std(random_acc_diff_all_pairs):.3f})"
    if len(random_acc_long_list) > 1:
      fancy_results[
          "P-value (Random)"] = f"{pval:.4f}" if pval >= 0.0001 else "< 0.0001"

  # prompt and response length as well as run time
  output_column_name = ["len_prompt", "len_response"]
  if "len_prompt_random" in performs_all_splits[0].keys():
    output_column_name += ["len_prompt_random", "len_response_random"]
  output_column_name += ["run_time"]
  if "run_time_random" in performs_all_splits[0].keys():
    output_column_name += ["run_time_random"]
  for name in output_column_name:
    perform_name = [perform[name] for perform in performs_all_splits]
    print(name.upper())
    print('> split_results:', ', '.join(map("{:.3f}".format, perform_name)))
    print(f"> mean: {np.mean(perform_name):.3f}")
    if len(perform_name) > 1:
      print(f"> std: {np.std(perform_name):.3f}")
    print("*-" * 50)
    print()
    str_format = ', '.join([f"{num:.3f}" for num in perform_name])
    str_format += f' | {np.mean(perform_name):.3f} ({np.std(perform_name):.3f})'
    fancy_results[NAME_MAP[name]] = str_format

  return fancy_results
