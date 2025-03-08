# Summarization
DATASET_DIR=./datasets/formatted_datasets/summarize/data.summarize.2025_03_05.jsonl
DATASET_ID=summarize
TEMPLATES="['rafailov-2023_summarize']"

# HHRLHF-Helpful
#DATASET_ID=hhrlhf_helpful
#TEMPLATES="['rafailov-2023_hhrlhf_helpful']"

MODEL_NAMES="['qwen25_32b_instr']"
API_KEY=http://10.10.10.132:8000/v1

SPLIT_SIZE=200
NUM_SPLITS=5
SELF_CONSIST_ID=0
NUM_RUNS=5
NUM_EVAL=-1
EXTRACT_RULE=combine
TEMPERATURE=0.1
NUM_WORKERS=4
CACHE_DIR=./outputs/

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
