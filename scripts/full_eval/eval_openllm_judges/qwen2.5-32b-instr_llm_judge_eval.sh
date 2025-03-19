MODEL_NAMES="['qwen2_5_32b_instr']"
API_KEY=http://10.10.10.132:8001/v1

SPLIT_SIZE=200
NUM_SPLITS=5
SELF_CONSIST_ID=0
NUM_RUNS=5
NUM_EVAL=-1
EXTRACT_RULE=combine
TEMPERATURE=0.1
NUM_WORKERS=8
CACHE_DIR=./outputs/

# Summarization
DATASET_DIR=./datasets/formatted_datasets/summarize/data.summarize.2025_03_05.jsonl
DATASET_ID=summarize
# TEMPLATES="['rafailov-2023_summarize']"
TEMPLATES="['chen-2023_summarize', 'guo-2024_summarize', 'liusie-2024_summarize', 'rafailov-2023_summarize', 'scheurer-2024_summarize', 'wang-2024_summarize', 'wu-2024_summarize', 'zheng-2023_summarize']"

python eval/eval_open_llm_judges.py \
--processed_data_path $DATASET_DIR \
--dataset_id $DATASET_ID \
--split_size $SPLIT_SIZE \
--num_splits $NUM_SPLITS \
--self_consist_id $SELF_CONSIST_ID \
--num_runs $NUM_RUNS \
--num_eval $NUM_EVAL \
--models "$MODEL_NAMES" \
--templates "$TEMPLATES" \
--extract_rule $EXTRACT_RULE \
--temperature $TEMPERATURE \
--num_workers $NUM_WORKERS \
--use_cache_results \
--use_cache_samples \
--cache_dir $CACHE_DIR \
--api_key $API_KEY

# HHRLHF-Helpful
DATASET_DIR=./datasets/formatted_datasets/hhrlhf_helpful/data.hhrlhf_helpful.2025_03_05.jsonl
DATASET_ID=hhrlhf_helpful
TEMPLATES="['bai-2023_hhrlhf_helpful', 'cheng-2024_hhrlhf_helpful', 'guo-2024_hhrlhf_helpful', 'mehta-2023_hhrlhf_helpful', 'rafailov-2023_hhrlhf_helpful', 'shen-2024_hhrlhf_helpful', 'wu-2023_hhrlhf_helpful', 'xu-2024_hhrlhf_helpful', 'zeng-2024_hhrlhf_helpful', 'zheng-2023_hhrlhf_helpful']"
# TEMPLATES="['rafailov-2023_hhrlhf_helpful']"
python eval/eval_open_llm_judges.py \
--processed_data_path $DATASET_DIR \
--dataset_id $DATASET_ID \
--split_size $SPLIT_SIZE \
--num_splits $NUM_SPLITS \
--self_consist_id $SELF_CONSIST_ID \
--num_runs $NUM_RUNS \
--num_eval $NUM_EVAL \
--models "$MODEL_NAMES" \
--templates "$TEMPLATES" \
--extract_rule $EXTRACT_RULE \
--temperature $TEMPERATURE \
--num_workers $NUM_WORKERS \
--use_cache_results \
--use_cache_samples \
--cache_dir $CACHE_DIR \
--api_key $API_KEY