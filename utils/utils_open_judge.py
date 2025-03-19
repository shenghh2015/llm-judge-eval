import os
import copy
import time
import regex
import importlib
from collections import defaultdict
# import openai
from openai import OpenAI

from configs.openai_api_key import OPENAI_MODELS

def import_template(template_name=''):
  """
  Import the dictionary with template, text process function, 
  regex pattern and matched character for template_name.
  """
  paper_name = template_name.split('_')[0]
  task = '_'.join(template_name.split('_')[1:])
  
  module = importlib.import_module(f".{template_name}", package=f"templates.{task}.{paper_name}")
  template = getattr(module, "template")

  return template

def remove_redundant(datalist):
  '''Remove redundant samples from the list of data.'''
  _datacache = defaultdict()
  for _data in datalist:
    prompt = _data["conversations"][0]['value']
    chosen = _data["chosen"]
    reject = _data["reject"]
    if not prompt + chosen + reject in _datacache:
      _datacache[prompt + chosen + reject] = _data
  return list(_datacache.values())

def create_comparison_prompt(original_text,
                             response1,
                             response2,
                             prompt):
  '''Create the comparison evaluation prompt.'''
  prompt = prompt["template"].format(original_text, response1, response2)
  return prompt

def evaluate_response_pair(original_text,
                           response1,
                           response2,
                           template,
                           model='gpt-4o',
                           temperature=0.1,
                           api_key=None,
                           max_tokens=2048,
                           ):
  '''Evaluate response pair (response1, response2) using the LLM judge.'''
  prompt = create_comparison_prompt(original_text, response1, response2, template)
  start_time = time.time()
  if model in OPENAI_MODELS:
    client = OpenAI(api_key=api_key)  # openai LLM
  else:
    client = OpenAI(api_key="EMPTY", base_url=api_key)   # open-sourced LLM
  judge_response = client.chat.completions.create(model=model,
                                                messages=[{
                                                    "role":
                                                    "system",
                                                    "content":
                                                    "You are a helpful assistant."
                                                }, {
                                                    "role": "user",
                                                    "content": prompt
                                                }],
                                                temperature=temperature,
                                                max_tokens=max_tokens)
  process_time = time.time() - start_time
  return judge_response.choices[0].message.content, len(prompt), len(
      judge_response.choices[0].message.content), process_time
  
def format_analysis_result(text_result, answer1, answer2, chosen_score,
                           reject_score):
  formated = f'''response A: {answer1}\n\nresponse B: {answer2}\n\nGPT-4 analysis: {text_result}\n\nchosen_score: {chosen_score}\n\nreject_score: {reject_score}\
                '''
  return formated

def get_rm_scores(text_results, template):
  '''Convert the text results into binary decisions for each response.''' 
  pattern = template["regex_pattern"]
  text_results = template["text_func"](text_results)
  match = regex.search(pattern, text_results)
  score = (match.group(1) == template["matched_char"]) * 1 if match else -1
  if score > -1:
    return score, 1 - score
  else:
    return -1, -1

def make_judge_result(_data,
                      template_name,
                      model='gpt-4o',
                      temperature=0.7,
                      mode='chosen_reject',
                      api_key=None,
                      data_idx=None,
                      results=[]):
  '''Get the LLM judging results for each data sample.'''
  
  # Get the template dictionary
  template = import_template(template_name)
  
  # Read data
  data = copy.deepcopy(_data)
  prompt = data["conversations"][0]['value']
  chosen = data["chosen"]
  reject = data["reject"]
  if "chosen_score" in data: data.pop("chosen_score")
  if "reject_score" in data: data.pop("reject_score")
  
  # Get text judging results from LLM outputs
  if mode == 'combine':
    _cached_result_cr, _len_prompt_cr, _len_response_cr, _run_time_cr = evaluate_response_pair(prompt, chosen, reject, template, 
                                                                                              model, temperature, api_key)
    _cached_result_rc, _len_prompt_rc, _len_response_rc, _run_time_rc = evaluate_response_pair(prompt, reject, chosen, template,
                                                                                              model, temperature, api_key)
    
    _cached_result = [_cached_result_cr, _cached_result_rc]
    _len_prompt = [_len_prompt_cr, _len_prompt_rc]  # length of the prompt text
    _len_response = [_len_response_cr,
                     _len_response_rc]  # length of the response text
    _run_time = [_run_time_cr, _run_time_rc]  # response time of one prompt
  
  elif mode == 'chosen_reject':
    _cached_result, _len_prompt, _len_response, _run_time = evaluate_response_pair(prompt, chosen, reject, template,
                                                                                  model, temperature, api_key)
  else:
    _cached_result, _len_prompt, _len_response, _run_time = evaluate_response_pair(prompt, reject, chosen, template,
                                                                                  model, temperature, api_key)

  if mode == 'combine':
    chosen_score1, reject_score1 = get_rm_scores(_cached_result[0], template)
    reject_score2, chosen_score2 = get_rm_scores(_cached_result[1], template)

    data['chosen_score'], data[
        'reject_score'] = chosen_score1 + chosen_score2, reject_score1 + reject_score2
    data['gpt4_eval'] = format_analysis_result(_cached_result[0], chosen+'(chosen)', reject+'(reject)', chosen_score1, reject_score1) + '\n'\
                        + format_analysis_result(_cached_result[1], reject+'(reject)', chosen+'(chosen)', chosen_score2, reject_score2)
    data['gpt4_eval_A'] = format_analysis_result(_cached_result[0],
                                                 chosen + '(chosen)',
                                                 reject + '(reject)',
                                                 chosen_score1, reject_score1)
    data['gpt4_eval_B'] = format_analysis_result(_cached_result[1],
                                                 reject + '(reject)',
                                                 chosen + '(chosen)',
                                                 chosen_score2, reject_score2)
    data['chosen_score_A'] = chosen_score1
    data['chosen_score_B'] = chosen_score2
    data['reject_score_A'] = reject_score1
    data['reject_score_B'] = reject_score2

    data['length_prompt_A'] = _len_prompt[0]
    data['length_prompt_B'] = _len_prompt[1]
    data['length_response_A'] = _len_response[0]
    data['length_response_B'] = _len_response[1]

    data['run_time_A'] = _run_time[0]
    data['run_time_B'] = _run_time[1]

  elif mode == 'chosen_reject':
    data['chosen_score'], data['reject_score'] = get_rm_scores(_cached_result, template)
    data['gpt4_eval'] = format_analysis_result(_cached_result,
                                               chosen + '(chosen)',
                                               reject + 'reject',
                                               data['chosen_score'],
                                               data['reject_score'])
    data['length_prompt'] = _len_prompt
    data['length_response'] = _len_response
    data['run_time'] = _run_time

  else:
    data['reject_score'], data['chosen_score'] = get_rm_scores(_cached_result, template)
    data['gpt4_eval'] = format_analysis_result(_cached_result,
                                               reject + '(reject)',
                                               chosen + '(chosen)',
                                               data['chosen_score'],
                                               data['reject_score'])
    data['length_prompt'] = _len_prompt
    data['length_response'] = _len_response
    data['run_time'] = _run_time
  
  results[data_idx] = data
  #return data
