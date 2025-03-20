from simple_eval.eval_utils import (jsonl_file_write,
                                    extract_json_from_text,
                                    compute_acc_both,
                                    compute_acc_random)
from utils.utils_read_write import jsonl_file_read
from collections import Counter
import os

def parse_datalist(datalist):
    
    unswapped = [int(extract_json_from_text(data["judging_text"]).get("preferred", "-1")) for data in datalist]
    swapped = [int(extract_json_from_text(data["judging_text_swapped"]).get("preferred", "-1")) for data in datalist]
    # unswapped_counts = Counter(unswapped)
    # swapped_counts = Counter(swapped)
    # print(swapped_counts, unswapped_counts)
    # print(unswapped[:10], swapped[:10])
    return unswapped, swapped

def main():
    # result_path = "outputs/.cache_eval_results/llama3_3_70b_instr_summary_judging_result.jsonl"
    # result_path = "outputs/.cache_eval_results/qwen2_5_32b_instr_summary_judging_result.jsonl"
    result_dir = "outputs/.cache_eval_results"
    file_names = [file_name for file_name in os.listdir(result_dir) if file_name.endswith("judging_result.jsonl")]
    
    for file_name in file_names:
        result_path = os.path.join(result_dir, file_name)
        # print(result_path)
        datalist = list(jsonl_file_read(result_path))
        unswapped, swapped = parse_datalist(datalist)

        acc_both, valid_both = compute_acc_both(unswapped, swapped)
        acc_random, valid_random = compute_acc_random(unswapped, swapped)
        acc_both, acc_random = round(acc_both, 3), round(acc_random, 3)
        print("--" * 50)
        print(file_name)
        print(f"acc_both: {acc_both}, valid_both: {valid_both} acc_random: {acc_random}, valid_random: {valid_random}")
        print()
        
        eval_result_path = result_path.replace("judging_result.jsonl", "eval_result.jsonl")
        if os.path.exists(eval_result_path):
            eval_results = list(jsonl_file_read(eval_result_path))
            eval_result = eval_results[0]
            eval_result["acc_both"] = round(acc_both, 3)
            eval_result["acc_random"] = round(acc_random, 3)
            eval_result["valid_count_both"] = round(valid_both)
            eval_result["valid_count_random"] = round(valid_random)
            jsonl_file_write([eval_result], eval_result_path)
    
if __name__ == "__main__":
    main()