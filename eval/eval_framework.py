""" 
The evaluation framework that connects four modules: data sampler, judges, metrics computation, and metrics visualization.
It allows a holistic flow from data to the evaluaton results, including metric report and visualization.   
"""
import os
import glob
from configs.openai_api_key import OPENAI_API_KEY, OPENAI_MODELS
from eval.data_sampler import DataSampler
from eval.judges import LLMJudge
from eval.metrics_computation import MetricsComputation
from eval.metrics_visualization import Visualization

""" The structure of the cached data:

    - datasets
        - raw_datasets
            - summarize
            - hhrlhf
        - formated_datasets
            - summarize
            - hhrlhf
    
    - outputs
        - dataset_id1, i.e. summarize
        
            - sampled_datasets
                - sampled_data.dataset_id1.date
                    - data.split_0.jsonl
                    - data.split_1.jsonl
                    - ...

            - judging_results
                - judge_name1
                    - date.dataset_id1.judge_name.split_id0.run1.jsonl   # flipping noise
                    - date.dataset_id1.judge_name.split_id0.run2.jsonl   # flipping noise
                    - date.dataset_id1.judge_name.split_id1.run1.jsonl   # other metrics
                    - date.dataset_id1.judge_name.split_id2.run1.jsonl   # other metrics
                    ...    
                ...
                
            - metrics_results
                - judge_name1
                    - date.acc_bias_results.dataset_id1.judge_name.jsonl  # accuracy and bias results
                    - date.self_consist_results.dataset_id1.judge_name.jsonl  # self-consistency (flipping noise) results
                ...
                
            - visualize_results
                - date
                    - meta_info.jsonl
                    - viz_results.dataset_id1.accuracy_both.png
                    - viz_results.dataset_id1.accuracy_random.png
                    - viz_results.dataset_id1.position_bias.png
                    - viz_results.dataset_id1.length_bias.png
                    - viz_results.dataset_id1.position_bias_accuracy.png
                    - viz_results.dataset_id1.length_bias_accuracy.png  
        ... 
"""

class JudgesEval:
    def __init__(self, data_path = "./datasets/formatted_datasets",
                       dataset_id = "summarize",
                       split_size = 200,
                       num_splits = 5,
                       self_consist_id = 0,
                       num_runs = 5,
                       num_eval = -1,
                       models = [],
                       templates = [],
                       extract_rule = "combine",
                       temperature = 0.1,
                       num_workers=8,
                       use_cache_samples=True,
                       use_cache_results=True,
                       cache_dir = "./outputs/",
                       api_key = None,
                ):
        """
        Evaluation framework:
        1. Data Sampler.
        2. LLM Judges.
        3. Metrics Computation.
        4. Metrics Visualization

        Args:
          - data_path: path to the preprocessed dataset.
          - dataset_id: data task (e.g. summarize, hhrlhf_helpful). Defaults to summarize.
          - split_size: number of samples in each data split. Defaults to 200.
          - num_splits: number of sampled splits. Defaults to 5.
          - self_consist_id: split index to make judging repeatedly in order to compute self-consistency results.
          - num_runs: number of runs for self_consist_split to compute self-consistency results. Default to 5.
          - num_eval: number samples to make judging for each split. Default to -1, which uses all the samples.
          - models: list of tested LLM names.
          - templates: list of tested templates.
          - extract_rule: rule to make LLM judge results to binary decision results 
                          ["combine", "chosen_reject", "reject_chosen"]. Defaults to "combine".
          - num_workers: the number of cpus to work in parallel.
          - use_cache_samples: if use cached samples (from the repeated stratified sampling) from the preprocessed dataset.
          - use_cache_results: if use cached results (if yes and the cached results are matched, the most recent cached results will be used).
          - cache_dir: directory to store the cached results. Defaults to "./outputs/".
          - api_key: api_key is explicitly specified while open-sourced LLM is used with vllm inference.
        """
        
        self.data_path = data_path
        self.dataset_id = dataset_id
        self.split_size = split_size
        self.num_splits = num_splits
        self.self_consist_id = self_consist_id
        self.num_runs = num_runs
        self.num_eval = num_eval
        self.models = models
        self.templates = templates
        self.extract_rule = extract_rule
        self.temperature = temperature
        self.num_workers = num_workers
        self.use_cache_samples = use_cache_samples
        self.use_cache_results = use_cache_results
        self.cache_dir = cache_dir
        self.api_key = OPENAI_API_KEY if api_key is None else api_key
        print("api_key: ", self.api_key)
        self.data_sampler = DataSampler(dataset_id = dataset_id,
                                        dataset_path = data_path,
                                        split_size = split_size,
                                        num_splits = num_splits,
                                        cache_dir = cache_dir,
                                        reset=(not self.use_cache_samples))
       
        print(">> data sampling is done!")
        self.judge_list = self._create_judges()
        self.metrics_computer = MetricsComputation(data_path=cache_dir, 
                                                  num_runs=num_runs, 
                                                  cache_dir=cache_dir)
        self.visualizer = Visualization(dataset_id=dataset_id, 
                                        cache_dir=cache_dir)
    
    def _check_models(self):
        '''Check if LLM names are valid.'''
        if len(self.models)==0: raise ValueError("model list cannot be empty!")
    
    def _check_templates(self):
        '''Check if templates are valid and implemented.'''
        if len(self.templates)==0: raise ValueError("template list cannot be empty!") 
        for template in self.templates:
            # Check if the template is corresponding to the data task
            paper = template.split("_")[0]
            task = '_'.join(template.split("_")[1:]) 
            if task!=self.dataset_id: 
                raise NameError(f"prompt template {template} is not for the task {self.dataset_id}!")
            # Check if the template is implemented
            if not os.path.exists(f"templates/{self.dataset_id}/{paper}/{template}.py"): 
                raise ValueError(f"{template} is not implemented!")
                
    def _create_judges(self):
        '''Build a set of judge eval processes.'''
        # Check model list and template list
        self._check_models()
        self._check_templates()
        
        return [
                LLMJudge(llm=llm, 
                         template=template, 
                         extract_rule=self.extract_rule, 
                         temperature=self.temperature, 
                         api_key=self.api_key,
                         data_path=self.cache_dir,
                         dataset_id=self.dataset_id,
                         cache_dir=self.cache_dir) 
              
                for llm in self.models
                    for template in self.templates
            ]
    
    def evaluate_judges(self):
        print(">> Evaluating all judges ...")
        judges_acc_bias_results = []  # accuracy bias results for all judges
        judges_self_consist_results = []  # self-consistency results for all judges
        for ldx, llm in enumerate(self.models):
            for tdx, template in enumerate(self.templates):
                llm_judge = self.judge_list[ldx*len(self.templates)+tdx]
                judge_results_all_splits = {}
                for split_id in range(self.num_splits):
                    num_runs = self.num_runs if split_id==self.self_consist_id else 1
                    for run_id in range(num_runs):
                        # Get the data for the split
                        datalist = self.data_sampler.get_split(split_id)
                        # Make judgings
                        judge_results = llm_judge.make_judging(datalist=datalist, 
                                                               split_id=split_id, 
                                                               run_id=run_id, 
                                                               num_eval=self.num_eval, 
                                                               num_workers=self.num_workers, 
                                                               use_cache=self.use_cache_results)
                        if run_id == 0:
                            judge_results_all_splits[split_id] = judge_results
                # Compute accuracy bias results
                num_eval = self.num_eval if self.num_eval!=-1 else self.split_size
                acc_bias_results = self.metrics_computer.compute_acc_bias(judge_results=judge_results_all_splits, 
                                                                          dataset_id=self.dataset_id, 
                                                                          llm=llm, 
                                                                          template=template, 
                                                                          num_splits=self.num_splits, 
                                                                          num_eval=num_eval, 
                                                                          use_cache=self.use_cache_results)
                # Compute self-consistency results
                self_consist_results = self.metrics_computer.compute_self_consistency(dataset_id=self.dataset_id, 
                                                                                      llm=llm, 
                                                                                      template=template, 
                                                                                      split_id=self.self_consist_id, 
                                                                                      num_runs=self.num_runs, 
                                                                                      num_eval=num_eval,
                                                                                      use_cache=self.use_cache_results)
                # Store results
                judges_acc_bias_results.append(acc_bias_results)
                judges_self_consist_results.append(self_consist_results)
                print(f"Finish the evaluation of LLM-Judge ({llm}, {template})!")
        
        # Post process results from all judges  
        all_results = self.metrics_computer.post_process_results(judges_acc_bias_results, 
                                                                 judges_self_consist_results,
                                                                 self.dataset_id, 
                                                                 llms=self.models, 
                                                                 templates=self.templates, 
                                                                 num_eval=num_eval,
                                                                 num_splits=self.num_splits, 
                                                                 num_runs=self.num_runs, 
                                                                 split_id=self.self_consist_id,
                                                                 use_cache=self.use_cache_results)
        
        # Visualize Accuracy (Both), Accuracy (Random), Position Bias and Length Bias
        self.visualizer.visualize_metrics(df_results=all_results, 
                                          llms=self.models, 
                                          templates=self.templates, 
                                          num_eval=num_eval, 
                                          num_splits=self.num_splits, 
                                          num_runs=self.num_runs, 
                                          split_id=self.self_consist_id, 
                                          use_cache=self.use_cache_results)

        print("** Finish evaluating all the judges **")
    
