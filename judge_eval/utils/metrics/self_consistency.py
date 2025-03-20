from judge_eval.utils.read_write import jsonl_file_read
import regex
import numpy as np


def get_p_consistent(filenames, swap_position, is_long=True):
  # Check the input
  assert swap_position in [
      "chosen-reject", "reject-chosen", "both"
  ], "swap position must be chosen-reject, reject-chosen, or both!"

  # Since in some splits some evaluation results are missing, we need to evaluate the self-consistency on the intersection
  all_convs = []
  for filename in filenames:
    conv_repeat = []
    results = jsonl_file_read(filename)
    for idx, data in enumerate(results):
      conv_repeat.append(''.join(
          [conv["value"] for conv in data["conversations"]]))
    all_convs.append(conv_repeat)

  # Get the intersection
  intersection = [
      conv for conv in all_convs[0]
      if all(conv in conv_repeat for conv_repeat in all_convs)
  ]

  # Get all the results
  chosen_scores_all = []
  for fid, filename in enumerate(filenames):
    chosen_scores_repeat = []
    results = jsonl_file_read(filename)
    for rid, data in enumerate(results):
      convs = ''.join([conv["value"] for conv in data["conversations"]])
      if convs not in intersection: continue
      # estimate p_consist of both chosen-reject and reject-chosen position
      if swap_position == "both":
        if is_long and (len(data["chosen"]) <= len(data["reject"])): continue
        if (not is_long) and (len(data["chosen"]) > len(data["reject"])):
          continue
        chosen_score = data["chosen_score"]
        assert chosen_score in [
            -2, -1, 0, 1, 2
        ], f"repeat {fid}, row {rid} has wrong chosen score: {chosen_score}!"
        if chosen_score == 2: chosen_score = 1  # choose only when [2,0]
        elif chosen_score == 1: chosen_score == 0  # ignore [1,1] case
        elif chosen_score == -2: chosen_score = -1
        else: chosen_score = chosen_score
        # estimate p_consist of chosen-reject or reject-chosen
      else:
        # get the chosen score
        column_name = "gpt4_eval_A" if swap_position == "chosen-reject" else "gpt4_eval_B"
        match = regex.search(r"chosen_score: (0|1|-1)", data[column_name])
        chosen_score = int(match.group(1))
        assert chosen_score in [
            -1, 0, 1
        ], f"repeat {fid}, row {rid} has wrong chosen score: {chosen_score}!"
      chosen_scores_repeat.append(chosen_score)
    chosen_scores_all.append(chosen_scores_repeat)

  # compute the consistent rate
  chosen_scores_all = np.array(chosen_scores_all)
  mask_chosen = (chosen_scores_all == 1).astype(int)
  mask_reject = (chosen_scores_all == 0).astype(int)
  mask = (chosen_scores_all != -1).astype(int)
  p_chosen = mask_chosen.sum(0) / (mask.sum(0) + 1e-16)
  p_reject = mask_reject.sum(0) / (mask.sum(0) + 1e-16)
  p_chosen_consist = (mask_chosen.sum(0) - 1) / (mask.sum(0) - 1 + 1e-16)
  p_reject_consist = (mask_reject.sum(0) - 1) / (mask.sum(0) - 1 + 1e-16)
  p_consist = (p_chosen * p_chosen_consist +
               p_reject * p_reject_consist).mean()

  if swap_position == "both":
    prefer = "long" if is_long else "others"
    return f"human prefer: {prefer} | p_consist: {p_consist:.3f}", p_consist
  else:
    return f"position: {swap_position} | p_consist: {p_consist:.3f}", p_consist


def evaluate_self_consistency(filenames, llm, template):

  # Self-Consistency Probability
  chosen_reject_result = get_p_consistent(filenames,
                                          swap_position="chosen-reject")

  reject_chosen_result = get_p_consistent(filenames,
                                          swap_position="reject-chosen")

  long_result = get_p_consistent(filenames, swap_position="both", is_long=True)

  short_result = get_p_consistent(filenames,
                                  swap_position="both",
                                  is_long=False)

  print("\n>>> Self Consistency Results <<<")
  print(chosen_reject_result[0])
  print(reject_chosen_result[0])
  print(long_result[0])
  print(short_result[0])

  # put results into a dictionary
  results = {
      "Model": llm,
      "Prompt": template,
      "p_consist_A": chosen_reject_result[1],
      "p_consist_B": reject_chosen_result[1],
      "p_consist_long": long_result[1],
      "p_consist_short": short_result[1]
  }

  return results
