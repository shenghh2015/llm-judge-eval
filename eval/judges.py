""" 
Judges used for pairwise LLM evaluation.
"""
import os
import glob
import regex
import copy
import time
from datetime import date
import multiprocessing
import openai

from configs.openai_api_key import OPENAI_MODELS
from utils.utils_read_write import jsonl_file_read, jsonl_file_write
from utils.utils_others import new_dir, find_latest
from utils.utils_judge import remove_redundant, make_judge_result


# Base class for all judge classes
class BaseJudge:
  def __init__(self, data_path, dataset_id, cache_dir):
    """
    Args:
      - data_path (str): path to the sampled data.
      - dataset_id (str): dataset name (e.g. summarize, hhrlhf-helpful).
      - judge_name (str): name of the judge.
      - cache_dir (str): directory to store all the cached judging results.
    """
    self.data_path = data_path
    self.dataset_id = dataset_id
    self.cache_dir = cache_dir
    self.judge_name = 'basejudge_template'
    
    assert os.path.exists(os.path.exists(os.path.join(self.data_path, self.dataset_id))), "sampled data path does not exist!"
  
  # private  
  def _init_judge(self):
    print('*' * 20, " Judge Info ", '*' * 20)
    print(f"Data: {self.dataset_id} | Judge: {self.judge_name}")
  
  def _check_cached_data(self, split_id, run_id, datalist_len):
    '''Check if the cached judging results exist for this judge.'''
    # Level 0: data path.
    # If multiple sampled version, the latest one will be returned.
    full_cache_dir = os.path.join(self.cache_dir, self.dataset_id, "judging_results", self.judge_name)
    latest = find_latest(os.path.join(full_cache_dir, f"**.{self.dataset_id}.{self.judge_name}.split_id{split_id}.run_id{run_id}.jsonl"))
    if latest == '': return False
    # Level 1: meta data info, such as size.
    if os.path.getsize(latest) == 0: return False
    # Level 2: data completeness
    cached_data = jsonl_file_read(latest)
    cached_datalist = [data["id"] for data in cached_data]
    if len(cached_datalist) != datalist_len: return False
    return True

  def _preprocess_data(self, datalist, num_eval=-1):
    # Print info
    print("*-" * 50)
    print(f'Process mode: {self.rule}')
    # Remove redundant samples
    datalist = remove_redundant(datalist)
    print(f'Number of data after redundancy reduction: {len(datalist)}')
    # Assign ids
    if not 'id' in datalist[0]:
      _datalist = []
      for i, _data in enumerate(datalist):
        _data['id'] = i
        _datalist.append(_data)
      datalist = _datalist
    # Get num_eval samples from datalist to make judging for
    if num_eval > 0: datalist = datalist[:num_eval]
    return datalist
  
  # public
  def make_judging(self, datalist):
    pass


class LLMJudge(BaseJudge): 
  def __init__(self, llm, template, extract_rule, temperature=0.1, api_key=None, *args, **kwargs):
    """
    Large Language Model Judge. Returns the binary judging results from LLM judge outputs.
    Args:
      - llm (str): name of large language model.
      - template (str): name of template used. 
      - extract_rule (str): rule to convert the output texts by LLM wto binary decisions (i.e. combine, chosen_reject, reject_chosen).
      - temperature (float, optional): temperature parameter of LLM. Defaults to 0.1.
      - num_workers (int, optional): number of works to process the data in parallel.
      - api_key (NoneType, optional): API key for commercial LLM. Defaults to None.
      - use_cache (bool, optional): if to use cached file. Defaults to True.
    """
    super(LLMJudge, self).__init__(*args, **kwargs)
    
    self.api_key = api_key
    
    if api_key is not None and llm in OPENAI_MODELS:
      print("Commercial LLM judge is used!")
      # self.api_key = api_key
    else:
      print("Open-sourced LLM judge is used!")
      
    assert extract_rule in ["combine", "chosen_reject", "reject_chosen"], f"extract_rule must be combine, chosen_reject or reject_chosen!"
    
    self.llm = llm          
    self.template_name = template     
    self.rule = extract_rule
    self.temperature = temperature
    self.judging_results = None
    
    # check the template to make sure it is corresponding to the correct dataset.
    if self.dataset_id == "summarize":
      assert self.template_name.split('_')[1]=="summarize", "template should be for summarization task!" 
    if self.dataset_id == "hhrlhf":
      assert self.template_name.split('_')[1]=="hhrlhf", "template should be for hh-rlhf task!" 
      
    # Print out judge information
    self.judge_name = self._init_judge()
  
  def _init_judge(self):
    '''Get the judge information.'''
    # Print out the judge information.
    print('*' * 20, " Judge Info ", '*' * 20)
    print(f"Data: {self.dataset_id} | LLM: {self.llm} | Template: {self.template_name} | Temperature: {self.temperature} | Extract Rule: {self.rule}")
    # Get the judge name to create the cached path.
    llm_name = regex.sub(r'[^a-zA-Z0-9]', '', self.llm)
    template_name = self.template_name.split('_')[0]  # remove the task
    template_name = regex.sub(r'[^a-zA-Z0-9]', '', template_name)
    return f"{llm_name}_{template_name}" # judge name
  
  def make_judging(self, datalist, split_id, run_id, num_eval=-1, num_workers=8, use_cache=True):
    """
    Make judging results for the list of data.
    Args:
      - datalist: list of sampled data.
      - split_id: split index.
      - run_id: run index. To measure flipping noise, there would be mulitple runs for the same split.
      - num_eval: number of evaluated samples. Defaults to -1, which means all the samples in the list.
      - num_workers: number of processes to process all the data in the list in parallel.
      - use_cache: use the cached judging results.
    """  
    # Preprocess the datalist
    datalist = self._preprocess_data(datalist, num_eval)
    full_cache_dir = os.path.join(self.cache_dir, self.dataset_id, "judging_results", self.judge_name)
    # If cached results exist, return the cached results
    if use_cache and self._check_cached_data(split_id, run_id, len(datalist)):
      latest = find_latest(os.path.join(full_cache_dir, f"**.{self.dataset_id}.{self.judge_name}.split_id{split_id}.run_id{run_id}.jsonl"))
      print(f">> Loading cached judging results from {latest} ...")
      cache_data = jsonl_file_read(latest)
      results = [data for data in cache_data]
      results = sorted(results, key=lambda x: x['id'])
      print(f'Number of judging results: {len(results)}')
      return results
    
    print(">> Making judging results ...")
    if self.api_key is not None: openai.api_key = self.api_key # set up api_key if commercial LLMs are used
    results = []
    with multiprocessing.Pool(num_workers) as pool:
      for data in datalist:
        pool.apply_async(make_judge_result,
                         args=(data, self.template_name, self.llm, self.temperature, self.rule, self.api_key),
                         callback=results.append)
      pool.close()
      pool.join()
    results = sorted(results, key=lambda x: x['id'])
    print(f'Number of judging results: {len(results)}')
    
    # Write the judging results into the file
    new_dir(full_cache_dir)  # if it does not exist, create the cache directory.
    output_path = os.path.join(full_cache_dir, f"{date.today().strftime('%Y_%m_%d')}.{self.dataset_id}.{self.judge_name}.split_id{split_id}.run_id{run_id}.jsonl")
    print(f">> Write judging results into {output_path} ...")
    jsonl_file_write(results, output_path)
    print("Done.")
    return results
  