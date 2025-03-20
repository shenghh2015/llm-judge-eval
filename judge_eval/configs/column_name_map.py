"""
Map between raw names and fancy names.
Raw names are used in evaluate_GPT4.py to store the results.
Fancy names are used to show the human-readable results in a .jsonl file.
"""

NAME_MAP = {
    "model": "Model",
    "prompt": "Prompt",
    "valid_count": "Valid Count",
    "acc": "Accuracy (Both)",
    "random_acc": "Accuracy (Random)",
    "valid_count_A": "Valid Count A",
    "valid_count_B": "Valid Count B",
    "acc_A": "Accuracy (A)",
    "acc_B": "Accuracy (B)",
    "acc_AB": "Accuracy (Mean)",
    "min_acc_AB": "Minimum Accuracy (Both)",
    "max_acc_AB": "Maximum Accuracy (Both)",
    "inconsistent_rate": "Inconsistent Rate",
    "acc_long": "Accuracy_long (Both)",
    "acc_not_long": "Accuracy_others (Both)",
    "random_acc_long": "Accuracy_long (Random)",
    "random_acc_not_long": "Accuracy_others (Random)",
    "len_prompt": "Prompt Length (Both)",
    "len_response": "Response Length (Both)",
    "len_prompt_random": "Prompt Length (Random)",
    "len_response_random": "Response Length (Random)",
    "run_time": "Run Time (Both)",
    "run_time_random": "Run Time (Random)"
}