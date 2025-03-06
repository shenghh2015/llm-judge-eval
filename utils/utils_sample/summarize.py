"""
Sampling functions for OpenAI Summarization dataset.
"""

import os
import argparse
from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedShuffleSplit
from utils.utils_read_write import jsonl_file_read


def sample_data(data_path, num_splits, split_size):
  """
  Args:
    - data_path: path to the preprocessed dataset.
    - num_splits: how many splits are sampled from the preprocessed dataset.
    - split_size: how many samples are contained in each split.
  
  Steps:
    1. For each post, randomly sample a chosen-reject pair.
    2. Stratified sampling based on whether human judges perfer longer or shorter (and the same).
    3. Repeat the step 2 for num_splits times, each split does not share any post with each other 
       in order to maximize the diversity of the dataset.
  """
  
  # Read in all the data
  # all the candidates must be in the test set
  data_list = jsonl_file_read(data_path)  # generator
  num_human_prefer_long = 0
  num_human_prefer_not_long = 0

  prompts = {}
  for idx, data in enumerate(data_list):
    
    if (data["split"] != "test") or (data["chosen"] == data["reject"]): continue

    convers = data['conversations']
    prompt_string = ''.join([c['value'] for c in convers])

    if len(prompt_string) == 0:  # prompt = ''
      prompt_string = data['chosen']

    label = int(len(data["chosen"]) > len(data["reject"]))
    num_human_prefer_long += label
    num_human_prefer_not_long += 1 - label
    
    # Store all the answer pairs from the same prompt
    if prompt_string in prompts:
      prompts[prompt_string].append((idx, label))
    else:
      prompts[prompt_string] = [(idx, label)]

  counts_per_prompt = [len(prompts[p]) for p in prompts]
  
  # Statistics
  count_mean = np.mean(counts_per_prompt).round(2)
  count_std = np.std(counts_per_prompt).round(2)
  count_max = np.max(counts_per_prompt)
  count_min = np.min(counts_per_prompt)

  print(
      f"> Total number of data: {num_human_prefer_long + num_human_prefer_not_long}"
  )
  print(f"> Num of unique prompts: {len(counts_per_prompt)}")
  print(
      f"> Num of data per prompt (mean/std/min/max): {count_mean}/{count_std}/{count_min}/{count_max}"
  )
  print(f"> Num of cases humans prefer long: {num_human_prefer_long}")
  print(f"> Num of cases humans prefer not long: {num_human_prefer_not_long}")

  # Sample data (for each prompt we only sample one answer pair)
  np.random.seed(1234)
  total_data = [
      answers[np.random.choice(len(answers))]
      for _, answers in prompts.items()
  ]
  
  num_long = 0
  num_not_long = 0
  for sample in total_data:
    num_long += sample[1]
    num_not_long += (1 - sample[1])
  print(
      f"> After pair sampling: # prefer long : # prefer short = {num_long} : {num_not_long} = {(num_long/num_not_long):.4f}"
  )

  # Stratefied sampling
  absolute_indeces = np.array([data[0] for data in total_data])
  preference_labels = np.array([data[1] for data in total_data])

  evaluation_indeces = {}
  print("stratefied sampling")
  for split_idx in range(num_splits):
    print(f"> split {split_idx}")
    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=split_size,
                                 random_state=1234)
    _, test_index = next(iter(sss.split(absolute_indeces, preference_labels)))
    print(test_index)
    print(preference_labels[test_index])
    num_prefer_long = preference_labels[test_index].sum()
    
    num_prefer_not_long = len(test_index) - num_prefer_long
    ratio = num_human_prefer_long / num_human_prefer_not_long
    print(
        f"> Stratified sampling: (split {split_idx}) # prefer long : # prefer not long = {num_prefer_long} : {num_prefer_not_long} = {ratio:.4f}"
    )
    evaluation_indeces[split_idx] = absolute_indeces[test_index].tolist()
    mask = np.isin(np.arange(len(absolute_indeces)), test_index, invert=True)
    absolute_indeces, preference_labels = absolute_indeces[
        mask], preference_labels[mask]  # remove the selected samples

  # Check correctness
  for i in range(num_splits):
    for j in np.arange(num_splits)[np.arange(num_splits) != i]:
      intersect_size = len(
          np.intersect1d(evaluation_indeces[i], evaluation_indeces[j]))
      assert intersect_size == 0, f"split {i} and {j} have shared sample!"

  # Output the data
  # get the mapping from absolute index to split index
  map_dict = {
      abs_index: split_idx
      for split_idx in range(num_splits)
      for abs_index in evaluation_indeces[split_idx]
  }
  # get all the selected indeces
  indeces_selected = list(map_dict.keys())
  # add split index to each selected item
  data_samples = []
  data_list = jsonl_file_read(data_path)  # generator

  for idx, data in enumerate(data_list):
    if idx in indeces_selected:
      data["eval_split"] = map_dict[idx]
      data_samples.append(data)
  
  return data_samples
