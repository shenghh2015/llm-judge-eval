import json
import random
import re


def jsonl_file_read(file_path: str) -> list:
  with open(file_path, 'r') as f:
    data_list = [json.loads(line) for line in f]
  return data_list


def json_file_write(datalist: list, file_path) -> None:
  with open(file_path, 'w') as f:
    for data in datalist:
      f.write(json.dumps(data) + '\n')


def prompting(prompt_template, prompt, response1, response2):
  return prompt_template.format(prompt, response1, response2)


def call_model_service(model_client, model, user_input, temperature,
                       max_new_tokens):
  judge_response = model_client.chat.completions.create(
      model=model,
      messages=[{
          "role": "system",
          "content": "You are a helpful assistant."
      }, {
          "role": "user",
          "content": user_input
      }],
      temperature=temperature,
      max_tokens=max_new_tokens)
  return judge_response.choices[0].message.content


def extract_json_from_text(text: str) -> dict:
  """
    Extracts JSON content from a text string and returns it as a dictionary.

    Args:
        text: The input text string.

    Returns:
        A dictionary representing the extracted JSON, or None if no JSON is found or if an error occurs during parsing.
  """
  json_matches = re.findall(r'\{[\s\S]*?\}', text)
  json_objects = []
  if json_matches:
    try:
      for json_str in json_matches:
        json_object = json.loads(json_str)
        if json_object is not None:
          json_objects.append(json_object)
      return json_objects[0] if len(json_objects) else {}
    except json.JSONDecodeError:
      print("Error: Invalid JSON format")
      return {}
  else:
    return {}


def compute_acc_both(outcome_list1, outcome_list2):
  valid_count = 0
  acc_both = 0
  assert len(outcome_list1) == len(outcome_list2), "Inconsistent lengths."

  for outcome1, outcome2 in zip(outcome_list1, outcome_list2):
    if outcome1 == -1 or outcome2 == -1:
      continue
    else:
      valid_count += 1
      if outcome1 == 1 - outcome2:
        acc_both += 1
  return acc_both / valid_count, valid_count


def compute_acc_random(outcome_list1, outcome_list2, random_seed):
  random.seed(random_seed)
  valid_count = 0
  acc_random = 0
  assert len(outcome_list1) == len(outcome_list2), "Inconsistent lengths."
  for outcome1, outcome2 in zip(outcome_list1, outcome_list2):
    outcome = outcome1 if random.random() < 0.5 else 1 - outcome2
    if outcome == -1:
      continue
    else:
      valid_count += 1
      acc_random += outcome
  return acc_random / valid_count, valid_count
