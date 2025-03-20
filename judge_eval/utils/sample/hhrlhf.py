"""
Sampling functions for HH-RLHF dataset.
"""
from sklearn.model_selection import StratifiedShuffleSplit
from judge_eval.utils.read_write import jsonl_file_read
import numpy as np
import regex


def get_prompt_string(data_path, is_uniturn, data):
  """
  Get the conversation based on the dataset and if it is single-turn.
  """
  if is_uniturn:
    # single-turn
    if (regex.search(r"full_hh_rlhf", data_path) is not None) and \
      (regex.search(r"\nAssistant:", data["conversations"][0]["value"]) is not None):
      return None
    if (regex.search(r"hh_rlhf_helpful|hh_rlhf_harmless", data_path) is not None) and \
      (len(data["conversations"]) > 1):
      return None
    prompt_string = data["conversations"][0]["value"]  # ignore "from"
  else:
    # multi-turn
    prompt_string = [
        conv["from"].capitalize() + ': ' + conv["value"].strip()
        for conv in data["conversations"]
    ]
    prompt_string = '\n'.join(prompt_string)
  return prompt_string


def process_data(data_path, is_uniturn):
  """
  Read and process the dataset.
  """
  # Check the data_path
  assert regex.search(
      r"hhrlhf_helpful|hhrlhf_harmless|full_hhrlhf",
      data_path) is not None, f"{data_path} is not for hh-rlhf dataset!"

  # Read the data
  hh_rlhf_data = jsonl_file_read(data_path)  # generator
  num_human_prefer_long = 0
  num_human_prefer_others = 0
  num_multiturn = 0
  num_uniturn = 0

  prompts = {}
  for idx, data in enumerate(hh_rlhf_data):
    # Skip invalid sample (empty chosen or reject answer, same chosen and reject answer)
    if (data["chosen"] == ' ') or (data["reject"] == ' '): continue
    if data["chosen"] == data["reject"]: continue

    # Process the dataset based on multi-turn or uni-turn
    prompt_string = get_prompt_string(data_path, is_uniturn, data)
    if prompt_string is None: continue

    # Get the label indicating if human prefers the longer answer
    prefer_label = (len(data["chosen"]) > len(data["reject"])) * 1
    num_human_prefer_long += prefer_label
    num_human_prefer_others += 1 - prefer_label

    # Get the label of multi-turn or single-turn conversation
    turn_label = (regex.search(r"\nAssistant:", prompt_string) is not None) * 1
    num_multiturn += turn_label
    num_uniturn += 1 - turn_label

    if prompt_string in prompts:
      prompts[prompt_string].append((idx, prefer_label, turn_label))
    else:
      prompts[prompt_string] = [(idx, prefer_label, turn_label)]

  counts_per_prompt = [len(prompts[p]) for p in prompts]
  # statistics
  count_mean = np.mean(counts_per_prompt).round(2)
  count_std = np.std(counts_per_prompt).round(2)
  count_max = np.max(counts_per_prompt)
  count_min = np.min(counts_per_prompt)

  print(
      f"> Total number of data: {num_human_prefer_long + num_human_prefer_others}"
  )
  print(f"> Num of unique prompts: {len(counts_per_prompt)}")
  print(
      f"> Num of data per prompt (mean/std/min/max): {count_mean}/{count_std}/{count_min}/{count_max}"
  )
  print(f"> Num of cases humans prefer long: {num_human_prefer_long}")
  print(f"> Num of cases humans prefer others: {num_human_prefer_others}")
  print(f"> Num of multi-turn conversations: {num_multiturn}")
  print(f"> Num of single-turn conversations: {num_uniturn}")

  return prompts


def sample_data(data_path, num_splits, num_samples, is_uniturn):
  """
  Randomly sample data stratified on the length preference by humans.

  Args:
    - data_path: path to the preprocessed dataset.
    - num_splits: number of total splits.
    - num_samples: num of samples in each split.
    - is_uniturn: if only single-turn conversations are sampled.
  
  Steps:
  1. For each conversation, randomly sample a chosen-reject pair
  2. Stratified sampling based on whether human judges prefer longer or shorter (and the same)
  3. Repeat the stratified sampling for num_splits times, each split does not share any post with each other 
     in order to maximize the diversity of the dataset.
  """
  # Process the data
  prompts = process_data(data_path, is_uniturn)

  np.random.seed(1234)
  total_data = [
      answers[np.random.choice(len(answers))]
      for _, answers in prompts.items()
  ]  # for each prompt we only sample one answer pair

  # Statistics of the preference mult-turn/single-turn ratio after sampling
  num_long = 0
  num_others = 0
  num_multiturn = 0
  num_uniturn = 0
  for sample in total_data:
    num_long += sample[1]
    num_others += (1 - sample[1])
    num_multiturn += sample[2]
    num_uniturn += (1 - sample[2])
  print(
      f"> After pair sampling: # prefer long : # prefer short = {num_long} : {num_others} = {(num_long/num_others):.4f}"
  )
  print(
      f"> After pair sampling: # multi-turn : # single-turn = {num_multiturn} : {num_uniturn} = {(num_multiturn/num_uniturn):.4f}"
  )

  # Stratefied sampling
  absolute_indeces = np.array([data[0] for data in total_data])
  preference_labels = np.array([data[1] for data in total_data])
  turn_labels = np.array([data[2] for data in total_data])

  evaluation_indeces = {}
  for split_idx in range(num_splits):
    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=num_samples,
                                 random_state=1234)
    _, test_index = next(iter(sss.split(absolute_indeces, preference_labels)))
    num_prefer_long = preference_labels[test_index].sum()
    num_prefer_others = len(test_index) - num_prefer_long
    ratio = num_prefer_long / num_prefer_others
    print(
        f"> Stratified sampling: (split {split_idx}) # prefer long : # prefer not long = {num_prefer_long} : {num_prefer_others} = {ratio:.4f}"
    )
    num_multiturn = turn_labels[test_index].sum()
    num_uniturn = len(test_index) - num_multiturn
    ratio = num_multiturn / num_uniturn
    print(
        f"> Stratified sampling: (split {split_idx}) # multi-turn : # single-turn = {num_multiturn} : {num_uniturn} = {ratio:.4f}"
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
  hh_rlhf_data = jsonl_file_read(data_path)  # generator

  num_multiturn = 0
  num_uniturn = 0
  for idx, data in enumerate(hh_rlhf_data):
    if idx in indeces_selected:
      data["eval_split"] = map_dict[idx]
      prompt_string = get_prompt_string(data_path, is_uniturn, data)
      assert prompt_string is not None, f"index {idx} is invalid!"
      if regex.search(r"\nAssistant:", prompt_string) is not None:
        num_multiturn += 1
      else:
        num_uniturn += 1
      if not is_uniturn:
        prompt_string += "\nAssistant: "  # if multi-turn, we add "Assistant:" to keep consistent with the LLM training.

      data["conversations"] = [{"value": prompt_string}]
      data["chosen"] = data["chosen"].strip()
      data["reject"] = data["reject"].strip()
      data_samples.append(data)

  return data_samples
