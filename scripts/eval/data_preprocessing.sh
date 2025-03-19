DATA_PATH=datasets/raw_datasets/
OUTPUT_DIR=datasets/formatted_datasets/
DATASET_ID=summarize   # summarize, hhrlhf_helpful
python datasets/data_preprocessing.py \
--data-path $DATA_PATH \
--output-dir $OUTPUT_DIR \
--dataset-id $DATASET_ID