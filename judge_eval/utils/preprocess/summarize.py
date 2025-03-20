"""
Data preprocess functions for OpenAI Summarization dataset.
"""
from tqdm import tqdm
"""
  rm_data = [
    {"conversations": [{
        "from":"human",
        "value": prompt
    }],
    "chosen": chosen_answer,
    "reject": reject_answer,
    "split": "train"/"test",
    "id": data id
    }
  ]
"""


def convert_to_rm_data(data, split):
  return {
      "id": data["id"],
      "conversations": [{
          "from": "human",
          "value": data["prompt"] + "\nTL;DR: "
      }],
      "chosen": data["chosen"].replace("TL;DR: ", ""),
      "reject": data["rejected"].replace("TL;DR: ", ""),
      "split": split
  }


def analyze_train_test(datalist):
  num_total, num_train, num_test = 0, 0, 0
  for _data in datalist:
    num_total += 1
    num_train += _data["split"] == "train"
    num_test += _data["split"] == 'test'
  print(f"total: {num_total}, train: {num_train}, test: {num_test}")


def process_data(loaded_dataset):
  splits = ['train', 'test']
  datalist = []
  for split in splits:
    split_dataset = loaded_dataset[split]
    for i, data in enumerate(tqdm(split_dataset)):
      data["id"] = i
      # skip the incorrect data
      rm_data = convert_to_rm_data(data, split)
      if len(rm_data["conversations"]) == 0 or len(
          rm_data["chosen"]) == 0 or len(rm_data["reject"]) == 0:
        continue
      datalist.append(convert_to_rm_data(data, split))
  analyze_train_test(datalist)
  return datalist
