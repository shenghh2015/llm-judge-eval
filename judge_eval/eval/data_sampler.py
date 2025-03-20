from judge_eval.utils.read_write import jsonl_file_read, jsonl_file_write
from judge_eval.utils.others import new_dir, find_latest
from judge_eval.utils.sample.summarize import sample_data as sample_summarize
from judge_eval.utils.sample.hhrlhf import sample_data as sample_hhrlhf
from datetime import date
import os
import glob


class Dataset:

  def __init__(self,
               dataset_id,
               dataset_path,
               split_size,
               num_splits,
               out_path,
               is_uniturn=False):
    """
    Args:
      - dataset_id: dataset name. (summarize: OpenAI summarization. 
                    hhrlhf_helpful: Anthropic hh-rlhf-helpful).
      - dataset_path: path to the preprocessed dataset.
      - split_size: the number of samples of each split.
      - num_splits: the number of splits. 
      - out_path: path to the stored sampled data.  
      - is_uniturn: if only conversations with single turn are kept for hh-rlhf dataset. 
    """
    self.dataset_id = dataset_id
    self.dataset_path = dataset_path
    self.split_size = split_size
    self.num_splits = num_splits
    self.out_path = out_path
    self.is_uniturn = is_uniturn

    # Samples from all splits
    self.data_splits = self.build_dataset()
    self.write_data()

  def _check_preprocessed_data(self):
    """
    Check if the preprocessed data exists.
    """
    # Level 0: data path
    if not os.path.exists(self.dataset_path):
      return False
    # Level 1: data size
    if os.path.getsize(self.dataset_path) == 0:
      return False
    return True

  def build_dataset(self):
    '''Build the dataset by sampling from the preprocessed dataset.'''
    if not self._check_preprocessed_data():
      raise ValueError(
          f"the preprocessed data does not exist! ({self.dataset_path})")
    # Sample data
    if self.dataset_id == "summarize":
      data_splits = sample_summarize(self.dataset_path, self.num_splits,
                                     self.split_size)
    elif self.dataset_id == "hhrlhf_helpful":
      data_splits = sample_hhrlhf(self.dataset_path, self.num_splits,
                                  self.split_size, self.is_uniturn)
    else:
      raise NameError(f"dataset {self.dataset_id} is not implemented!")
    return data_splits

  def write_data(self):
    '''Write the data into the output file'''
    out_path = os.path.join(
        self.out_path, self.dataset_id, "sampled_datasets",
        f"sampled_data.{self.dataset_id}.{date.today().strftime('%Y_%m_%d')}")
    new_dir(out_path)
    data_each_split = {split_id: [] for split_id in range(self.num_splits)}
    for data in self.data_splits:
      data_each_split[data["eval_split"]].append(data)
    for split_id in range(self.num_splits):
      jsonl_file_write(data_each_split[split_id],
                       os.path.join(out_path, f"data.split_{split_id}.jsonl"))
    print(
        f">> Finish writing the sampled data ({self.num_splits} splits) into the output {out_path}!"
    )


class DataSampler:
  """
  DataSampler is a data manager that take charge of
    1) data sampling
    2) data caching
    3) data fetching
  """

  def __init__(self,
               dataset_id,
               dataset_path,
               split_size,
               num_splits,
               cache_dir,
               is_uniturn=False,
               reset=False):
    """
    Args:
      - dataset_id: dataset to sample (e.g. summarize, hh-rlhf).
      - dataset_path: path to the preprocessed dataset.
      - split_size: the number of samples in each split.
      - num_splits: the number of splits in the sampled dataset.
      - cache_dir: path to the dir containing the cached sampled dataset.
      - is_uniturn: only conversations with single turn will be kept for hh-rlhf dataset.
      - reset: if True, build the dataset from scratch.
    """
    self.dataset_id = dataset_id
    self.dataset_path = dataset_path
    self.split_size = split_size
    self.num_splits = num_splits
    self.cache_dir = cache_dir
    self.is_uniturn = is_uniturn
    self.dataset = self.sampling(reset)

  def _check_cached_data(self):
    """ 
    Check if there is a valid cached dataset
    """
    # Level 0: data path.
    # If multiple sampled version, the latest one will be returned.
    latest = find_latest(
        os.path.join(self.cache_dir, self.dataset_id, "sampled_datasets",
                     f"sampled_data.{self.dataset_id}.**/"))
    if latest == '': return False
    # Level 1: meta data info, such as size.
    files_splits = sorted(
        glob.glob(os.path.join(latest, f"data.split_*.jsonl")))
    if len(files_splits) != self.num_splits: return False
    for file in files_splits:
      if os.path.getsize(file) == 0: return False
    # Level 2: data completeness
    num_data = 0
    for file in files_splits:
      data_split = jsonl_file_read(file)
      for data in data_split:
        num_data += 1
        if len(data["conversations"]) == 0 or len(data["chosen"]) == 0 or len(
            data["reject"]) == 0:
          return False
    if num_data != self.num_splits * self.split_size: return False
    return True

  def sampling(self, reset=False):
    if reset or (not self._check_cached_data()):
      print(">> Sampling the dataset ...")
      sampled_dataset = Dataset(self.dataset_id,
                                self.dataset_path,
                                self.split_size,
                                self.num_splits,
                                self.cache_dir,
                                is_uniturn=self.is_uniturn)
      sampled_dataset = sampled_dataset.data_splits
      dataset = {}
      for data in sampled_dataset:
        if data["eval_split"] not in dataset.keys():
          dataset[data["eval_split"]] = []
        dataset[data["eval_split"]].append(data)
      print(">> Finished!")
    else:
      # Load the cached data
      cached_folder = find_latest(
          os.path.join(self.cache_dir, self.dataset_id, "sampled_datasets",
                       f"sampled_data.{self.dataset_id}.**/"))
      print(f">> Loading the cached dataset from {cached_folder} ...")
      dataset = {}
      for split_id in range(self.num_splits):
        data_split = jsonl_file_read(
            os.path.join(cached_folder,
                         f"data.split_{split_id}.jsonl"))  # generator
        dataset[split_id] = [data for data in data_split]
      print("Finished!")
    return dataset

  def get_split(self, split_id):
    '''Get a split based on split_id'''
    assert (split_id >= 0) and (
        split_id < self.num_splits
    ), f"split_id should be in the range of [0, {self.num_splits - 1}]"
    return self.dataset[split_id]

  def get_splits(self, split_ids):
    '''Get splits whose ids are specified in split_ids'''
    assert (len(split_ids) > 0) and (len(set(split_ids)) == len(
        split_ids)), "split_ids must be non-empty and contain unique ids!"
    data_splits = []
    for split_id in split_ids:
      data_splits.extend(self.get_split(split_id))
    return data_splits
