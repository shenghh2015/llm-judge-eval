"""
Data preprocess functions for HH-RLHF dataset.
"""
import regex
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


def convert_to_rm_data(data):
  # Step 0: get chosen and reject
  chosen = data["chosen"]
  reject = data["rejected"]

  # Step 1: preprocess the chosen and the reject
  def _preprocess(convs):
    convs = regex.sub(r"\n+", "\n", convs)
    convs = convs.replace("\nHuman:", "\n\nHuman:")
    convs = convs.replace("\nAssistant:", "\n\nAssistant:")
    return convs

  chosen = _preprocess(chosen)
  reject = _preprocess(reject)
  # Step 1: get the conversations
  conv_list = chosen.strip().split("\n\n")
  conv_list = conv_list[:-1]
  # Step 2: convert the conversations into the format {"from":"Human", "value":' xxxx'}
  conv_hist = [{
      'from': conv.split(':')[0].strip().lower(),
      'value': conv.split(':')[1]
  } for conv in conv_list]
  # Step 3: get the chosen answer
  chosen = chosen.strip().split("\n\n")[-1].split(":")[-1]
  reject = reject.strip().split("\n\n")[-1].split(":")[-1]

  return {'conversations': conv_hist, 'chosen': chosen, 'reject': reject}


def process_origin_rm_data(dataset):
  splits = list(dataset.keys())
  datalist = []
  data_id = 0
  seen = set()
  for split in splits:
    split_dataset = dataset[split]
    for data in tqdm(split_dataset):
      rm_data = convert_to_rm_data(data)
      # skip the incorrect data
      if len(rm_data["conversations"]) == 0 or len(
          rm_data["chosen"]) == 0 or len(rm_data["reject"]) == 0:
        continue
      rm_data["split"] = split
      rm_data["id"] = data_id
      prompt_chosen_reject = rm_data["conversations"][0]["value"] + rm_data[
          "chosen"] + rm_data["reject"]
      if prompt_chosen_reject in seen: continue
      seen.add(prompt_chosen_reject)
      datalist.append(rm_data)
      data_id += 1

  return datalist


def analyze_train_test(datalist):
  num_train, num_test = 0, 0
  for _data in datalist:
    num_train += _data["split"] == "train"
    num_test += _data["split"] == 'test'
  print(f"train: {num_train}, test: {num_test}")


def process_data(loaded_dataset):

  # Convert to rm data format
  rm_datalist = process_origin_rm_data(loaded_dataset)

  analyze_train_test(rm_datalist)

  return rm_datalist
