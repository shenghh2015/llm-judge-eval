## Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks: Explainable Metrics and Diverse Prompt Templates
* Code coming soon.

### Example Evaluation Results
#### Metric Report Tables
Metric report tables related to evaluating LLM judges (``model:GPT-4o`` with different templates) on the ``TL;DR Summarization`` dataset.
<div style="display: grid; grid-template-columns: repeat(1, 1fr); gap: 2px; text-align: center;" >
  <div>
    <img src="./example_results/metrics_table.jpg" alt="Accuracy (Both)" style="width:70%;">
  </div>
</div>

#### Visualization Results
Visualization results related to evaluating LLM judges (models + different templates) on the ``TL;DR Summarization`` dataset.\
<img src="./example_results/accuracy_both_summary.png" width="300"/>   <img src="./example_results/position_bias_summary.png" width="300"/>
<img src="./example_results/length_bias_summary.png" width="300"/>   <img src="./example_results/position_bias_accuracy_summary.png" width="300"/>

### Citations
```
@article{wei2024systematic,
  title={Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks: Explainable Metrics and Diverse Prompt Templates},
  author={Wei, Hui and He, Shenghua and Xia, Tian and Wong, Andy and Lin, Jingyang and Han, Mei},
  journal={arXiv preprint arXiv:2408.13006},
  year={2024}
}
```