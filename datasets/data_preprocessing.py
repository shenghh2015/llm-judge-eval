from datasets import load_dataset, load_from_disk
from judge_eval.utils.others import new_dir, find_latest
from judge_eval.utils.read_write import jsonl_file_write
from judge_eval.utils.preprocess.summarize import process_data as preprocess_summarize
from judge_eval.utils.preprocess.hhrlhf import process_data as preprocess_hhrlhf
import os
import glob
import argparse
from datetime import date


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None, 
                        help="dataset downloaded from the original data source")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="directory to save the processed datasets")
    parser.add_argument("--dataset-id", type=str, default=None,
                        help="evaluation dataset (summarize, hhrlhf_helpful)")
    return parser.parse_args()


class DataProcessor:
  """
  File structure:
    -  datasets
        - raw_datasets
          - summarize
            - raw_data.summarize.date
          - hhrlhf_helpful
            - raw_data.hhrlhf_helpful.date
        
        - formatted_datasets
          - summarize
            - processed_data.summarize.date.jsonl
          - hhrlhf_helpful
            - processed_data.hhrlhf_helpful.date
  """
  def __init__(self, data_path, output_dir, dataset_id):
    """
    Args:
      - data_path: the path to the downloaded dataset. If there is no downloaded dataset, 
                   download it and store it here.
      - output_dir: the path to the preprocessed dataset.
      - dataset_id: dataset name. (summarize: OpenAI summarization. 
                    hh_rlhf_helpful: Anthropic hh-rlhf-helpful.)
    """
    
    self.data_path = data_path
    self.output_dir = output_dir
    self.dataset_id = dataset_id
    
    assert self.dataset_id in ["summarize", "hhrlhf_helpful"], f"{self.dataset_id} is not implemented!"
    
    # Construct folders if they do not exist
    new_dir(os.path.join(self.data_path, self.dataset_id))
    new_dir(os.path.join(self.output_dir, self.dataset_id))
    
  def _check_data(self, check_path, name_format, check_format):
    # Find the latest version
    latest = find_latest(os.path.join(check_path, self.dataset_id, name_format))
    # Level 0: data path
    if latest == '': return False
    # Level 1: data size
    if check_format == ".arrow":  # downloaded data
      checked_files = glob.glob(os.path.join(latest, "**", "**.arrow"))
      if len(checked_files) == 0: return False 
    elif check_format == ".jsonl":  # preprocessed data
      if os.path.getsize(latest) == 0: return False
    else: raise NameError(f"{check_format} is not included!")
    
    return True
  
  def download_dataset(self):
    '''Download the dataset if it does not exist.'''
    if self._check_data(self.data_path, f"raw_data.{self.dataset_id}.**/", ".arrow"):
      print(f"downloaded dataset for {self.dataset_id} already exists!")
      download_dataset = find_latest(os.path.join(self.data_path, self.dataset_id, f"raw_data.{self.dataset_id}.**/"))
      loaded_dataset = load_from_disk(download_dataset)
    else:
      if self.dataset_id == "summarize":
        loaded_dataset = load_dataset("CarperAI/openai_summarize_comparisons")
      elif self.dataset_id == "hhrlhf_helpful":
        loaded_dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
      else: 
        raise NameError(f"{self.dataset_id} is not implemented!")
      # Save it into the output folder
      out_name = f"raw_data.{self.dataset_id}.{date.today().strftime('%Y_%m_%d')}"    
      out_path = os.path.join(self.data_path, self.dataset_id, out_name)
      print(f"Save it to {out_path} ...")
      loaded_dataset.save_to_disk(out_path)
      print("Done.")
      
    return loaded_dataset

  def preprocess_dataset(self):
    # Check if the preprocessed dataset exists
    if self._check_data(self.output_dir, f"data.{self.dataset_id}.**.jsonl", ".jsonl"):
      print(f"processed dataset for {self.dataset_id} already exists!")
    else:
      # Download the dataset
      loaded_dataset = self.download_dataset()
      # Preprocess the data
      print(f"Preprocessing {self.dataset_id} data ...")
      if self.dataset_id == "summarize": 
        processed_dataset = preprocess_summarize(loaded_dataset)
      elif self.dataset_id == "hhrlhf_helpful":
        processed_dataset = preprocess_hhrlhf(loaded_dataset)
      else: 
        raise NameError(f"{self.dataset_id} is not implemented!")
      # Write the data into output_dir
      out_path = os.path.join(self.output_dir, self.dataset_id, f"data.{self.dataset_id}.{date.today().strftime('%Y_%m_%d')}.jsonl")
      jsonl_file_write(processed_dataset, out_path)
      
    
def main():
  # Get arguments
  args = get_args()
  # Initiate a preprocessor
  data_processor = DataProcessor(args.data_path, args.output_dir, args.dataset_id)
  # Preprocess the original dataset
  data_processor.preprocess_dataset()
  # data_processor.download_dataset()
  

if __name__ == "__main__":
  main()
  