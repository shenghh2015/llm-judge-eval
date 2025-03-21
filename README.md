<h1 align="center">Systematic Evaluation of LLM judges for Pairwise Evaluation: Explainable Metrics, Diverse Prompt Templates, and Diverse Models</h2>

[📖 `ICRL 2025 Workshop Paper` ](https://arxiv.org/abs/2408.13006) 
[🚀 `Quick Evaluation Tool`](#llms-as-judges-quick-evaluation)
[📜 `BibTeX` ](#references)

## Table of Contents
We aim at providing `Judge Evaluation Framework (JudgeEval)` for _quickly_ and _conveniently_ examinating the ability (accuracy_both, accuracy_random) and vunerabilities (position bias, length bias, and self-inconsisitency) of LLM-as-a-judge methods for diverse taks. Our repository includes:
* [🚀 `Quick Evaluation Tool`](#llms-as-judges-quick-evaluation): A simple evaluation tool that allows users to _quickly_ examine the judging ability of `commercial models` (OpenAI models) and `open-source models` with minimal setup effort.
* [📖`ICRL 2025 Workshop Paper`](https://arxiv.org/abs/2408.13006): A codebase for our paper: _Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks: Explainable Metrics and Diverse Prompt Templates_. We systematically evaluated LLM-as-a-Judge methodolodies on two datasets (i.e ``TL;DR Summerization`` and ``HH-RLHF-Helpful``):
  - We define evaluation metrics with improved theoretical interpretability. 
  - We develop a framework to evaluate, compare, and visualize the reliability and alignment of LLM judges.
  - We investigate the effect of diverse prompt templates on LLM-judge reliability. 
  - Our results indicate a significant impact of prompt templates on LLM judge performance, as well as a mediocre alignment level between the tested LLM judges and human evaluators.

<div align="center">
  <img src="./examples/example_results/framework.jpg" width="80%"/>
</div>
<!-- ![Evaluation Framework](./examples/example_results/framework.jpg) -->

## 🛠️ Installation
### Option 1: Using `uv` (highly recommended)
```bash
# install uv
pip install uv

# build virtual environment and activate the virtual environment
uv venv --python 3.11
source .venv/bin/activate

# install requirement
uv pip install -r requirements.txt

# export python path
vim ~/.bashrc
export PYTHONPATH=./
source ~/.bashrc
```

### Option 2: Using conda installation
```bash
conda -n llm_judge_eval python=3.11
conda activate llm_judge_eval
pip install -r requirements.txt

# export python path
vim ~/.bashrc 
export PYTHONPATH=.
source ~/.bashrc
```

## 🚀 LLMs-as-Judges Quick Evaluation

### OpenAI GPT models
```bash
export OPENAI_API_KEY=[YOUR OPENAI API KEY]  # You need to set OPENAI_API_KEY to run openai models for judging
cd llm-judge-eval
python simple_eval/judge_eval.py --model gpt-4o-mini --task summary            # summary task
python simple_eval/judge_eval.py --model gpt-4o-mini --task hh_rlhf_helpful    # hh_rlhf_helpful task
python simple_eval/judge_eval.py --model gpt-4o --task summary                 # summary task
python simple_eval/judge_eval.py --model gpt-4o --task summhh_rlhf_helpfulary  # summary task
```

### Open-sourced models
#### Set up an LLM inference API service using [vllm](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
```bash
# An example of setting up an LLM inference API service
CUDA_VISIBLE_GPUS=0 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --served-model-name qwen2.5_7b_instruct --max-model-len 2024
```
The inference API service set up in the example is compatible with the OpenAI API. It has a model name of `qwen2.5_7b_instruct` and base_url of `http://localhost:8000/v1`. You can use the following commands to check whether the service is set up successfully or not.
```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5_7b_instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

#### Run LLM judge evaluation
```bash
# enter the project directory
cd llm-judge-eval

# summary task
python simple_eval/judge_eval.py --model qwen2.5_7b_instruct --base_url http://localhost:8000/v1 --task summary
# {'acc_both': 0.58, 'valid_count_both': 200, 'acc_random': 0.645, 'valid_count_random': 200}

# hh_rlhf_helpful task    
python simple_eval/judge_eval.py --model qwen2.5_7b_instruct --base_url http://localhost:8000/v1 --task hh_rlhf_helpful
```

### LLM-as-Judge evaluation on a mini dataset
We build a mini-benchmark dataset that contains `400` test cases: `200` are drawn from the `summarize` dataset, and the other 200 are drawn from the `hhrlhf_helpful` dataset, using a stratified sampling strategy introduced in our paper.
We adopted prompting templates (`simple_eval/templates`) that are similar to the ones suggested in [Rafailov et al.](https://arxiv.org/abs/2305.18290), as their templates are generally robust based on our systematic evaluation with diverse prompt templates.
We evaluated widely-used commercial models (`gpt-4o`) and open-source state-of-the-art models (`Qwen2.5`, `LLaMA3-3`, `DeepSeek-R1`). The results are shown below. Here, we aimed to provide a general comparison.

#### Evaluation Results

* Summary Task

| Models                 | Acc_both | Valid_count_both | Acc_random | Valid_count_random |
|------------------------|---------|------------------|------------|--------------------|
| GPT-4o                 | 0.645   | 200              | 0.705      | 200               |
| GPT-4o-mini            | 0.52    | 200              | 0.68       | 200               |
| Deeoseek-R1-70B        | 0.596   | **198**          | 0.707      | **198**           |
| Deepseek-R1-32B        | 0.482   | **195**          | 0.657      | **198**           |
| Qwen2.5-32B-Instruct   | 0.58    | 200              | 0.7        | 200               |
| Qwen2.5-72B-Instruct   | 0.563   | 199              | 0.72       | 200               |
| QwQ-32B                | 0.675   | **154**          | 0.705      | **176**           |
| LLaMA-3.1-70B-Instruct | 0.535   | 200              | 0.71       | 200               |
| LLaMA-3.3-70B-Instruct | 0.52    | 200              | 0.705      | 200               |

* HH-RLHF-Helpful task

| Models                 | Acc_both | Valid_count_both | Acc_random | Valid_count_random |
|------------------------|---------|------------------|------------|--------------------|
| GPT-4o                | 0.57    | 200              | 0.675      | 200               |
| GPT-4o-mini           | 0.34    | 200              | 0.65       | 200               |
| Deeoseek-R1-70B       | 0.52    | 200              | 0.66       | 200               |
| Deepseek-R1-32B       | 0.475   | 200              | 0.585      | 200               |
| Qwen2.5-32B-Instruct  | 0.485   | 200              | 0.625      | 200               |
| Qwen2.5-72B-Instruct  | 0.42    | 200              | 0.63       | 200               |
| QwQ-32B               | 0.617   | **133**          | 0.673      | **156**           |
| LLaMA-3.1-70B-Instruct| 0.5     | 200              | 0.65       | 200               |
| LLaMA-3.3-70B-Instruct| 0.3     | 200              | 0.565      | 200               |

---

### **Notes:**
- **Bold numbers** indicate values that are less than 200 in the `Valid_count_both` and `Valid_count_random` columns.
- *Acc_both* and *Acc_random* are the accuracies that measure the judging abilities of LLM judges.
- *Valid_count_both* and *Valid_count_radom* are the number of results among the 200 test cases that generate the required format, which rougly indicate the instruction following abilities.

## 📖 Systematic Evaluation of LLM-as-Judges
### Dataset Preprocessing
Use the following command to prepare a formatted dataset for the LLM judge evaluation process. 
The default dir to save the processed dataset ``./datasets/formatted_datasets``. 
The ``dataset_id`` identifies the formatted dataset, which is better kept consistent in the following steps.

```bash
python datasets/data_preprocessing.py \      
--data-path datasets/raw_datasets/ \         # directory to save downloaded dataset from the original data source
--output-dir datasets/formatted_datasets/ \  # directory to save the processed datasets
--dataset-id summarize                       # summarize, hhrlhf_helpful
```

### Add OpenAI Key
Add your own OpenAI key to ``configs/openai_api_key.py`` in order to evaluate LLM judges.

### Evaluate a Set of LLM Judges by Metric Computation and Visualization
Use the example below to evaluate a set of LLM judges using the example dataset ``dataset_id=summarize``.
The templates are specified in ``templates/dataset_id`` folders.

```bash
python eval/eval_llm_judges.py \
--processed_data_path ./datasets/formatted_datasets/summarize/data.summarize.xxxx-xx-xx.jsonl \  # data path to the preprocessed dataset
--dataset_id summarize \  # dataset task (summarize or hhrlhf-helpful)
--split_size 200 \        # number of samples in each split
--num_splits 5 \          # number of splits
--self_consist_id 0 \     # index of split used to compute self-consistency results
--num_runs 5 \            # number of repetition to run the split to compute the self-consistency results                                                                     
--num_eval -1 \           # number of evaluated samples in each split (-1 means all samples in the split)
--models "['gpt-4o-mini']" \  # list of LLM names
--templates "['chen-2023_summarize', 'guo-2024_summarize']" \  # list of templates
--extract_rule combine \  # rule to make binary output from the judging results ("combine", "chosen_reject" or "reject_chosen")
--temperature 0.1 \       # temperature parameter used for LLM inference
--num_workers 8 \         # number of processes to run judging results in parallel
--use_cache_samples \     # if use cached sampling results
--use_cache_results \     # if use cached computation and visualization results
--cache_dir ./outputs/    # directory to store the output results
```

### Example Evaluation Results
##### Metric Report Tables
Metric report tables related to evaluating LLM judges (``model:GPT-4o`` with different templates) on the ``TL;DR Summarization`` dataset.
<div style="display: grid; grid-template-columns: repeat(1, 1fr); gap: 2px; text-align: center;" >
  <div>
    <img src="./examples/example_results/metrics_table.jpg" alt="Accuracy (Both)" style="width:70%;">
  </div>
</div>

##### Visualization Results
Visualization results related to evaluating LLM judges (models + different templates) on the ``TL;DR Summarization`` dataset.\
<img src="./examples/example_results/accuracy_both_summary.png" width="300"/>   <img src="./examples/example_results/position_bias_summary.png" width="300"/>
<img src="./examples/example_results/length_bias_summary.png" width="300"/>   <img src="./examples/example_results/position_bias_accuracy_summary.png" width="300"/>

## 📜 References
If you find the code and processed datasets useful in your work, please consider citing the following paper:
```bibtex
@article{wei2024systematic,
  title={Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks: Explainable Metrics and Diverse Prompt Templates},
  author={Wei, Hui and He, Shenghua and Xia, Tian and Wong, Andy and Lin, Jingyang and Han, Mei},
  journal={arXiv preprint arXiv:2408.13006},
  year={2024}
}
```