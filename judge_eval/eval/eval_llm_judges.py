from judge_eval.eval.eval_framework import JudgesEval
import argparse


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--processed_data_path",
                      type=str,
                      default="./datasets/formatted_datasets",
                      help="Path to the preprocessed data.")

  parser.add_argument("--dataset_id",
                      type=str,
                      default="summarize",
                      help="data task (e.g. summarize, hhrlhf_helpful)")

  parser.add_argument("--split_size",
                      type=int,
                      default=200,
                      help="The number of samples in each evaluation split.")

  parser.add_argument("--num_splits",
                      type=int,
                      default=5,
                      help="The total number of evaluation splits.")

  parser.add_argument(
      "--self_consist_id",
      type=int,
      default=0,
      help=
      "The index of the split to compute self-consistency results (i.e. flipping noise)."
  )

  parser.add_argument(
      "--num_runs",
      type=int,
      default=5,
      help=
      "The number of repetition to run the split indexed by self_consist_id to measure the self-consistency."
  )

  parser.add_argument(
      "--num_eval",
      type=int,
      default=-1,
      help=
      "The number of samples to make judges for in each split (-1 means all samples)."
  )

  parser.add_argument("--models",
                      type=lambda x: eval(x),
                      default=[""],
                      help="The list of tested Large Language Model names.")

  parser.add_argument("--templates",
                      type=lambda x: eval(x),
                      default=[""],
                      help="The list of tested template names.")

  parser.add_argument(
      "--extract_rule",
      type=str,
      default="combine",
      help=
      "The rule to extract binary decisions from LLM judge results ('combine', 'chosen_reject' or 'reject_chosen')"
  )

  parser.add_argument(
      "--temperature",
      type=float,
      default=0.1,
      help="Temperature parameter for Large Language Model based Judges.")

  parser.add_argument(
      "--max_new_tokens",
      type=int,
      default=1024,
      help="Temperature parameter for Large Language Model based Judges.")
  
  parser.add_argument(
      "--num_workers",
      type=int,
      default=8,
      help="The number of processes to make judging results in parallel.")

  parser.add_argument("--use_cache_samples",
                      action="store_true",
                      help="If use cached sampling results.")

  parser.add_argument(
      "--use_cache_results",
      action="store_true",
      help=
      "If use cached judging, metrics computation and visualization results.")

  parser.add_argument("--cache_dir",
                      type=str,
                      default="./outputs/",
                      help="Directory to store cached and output results.")

  return parser.parse_args()


def main():
  # Get arguments
  args = get_args()
  print(args)

  # Initialize
  judges_evals = JudgesEval(data_path=args.processed_data_path,
                            dataset_id=args.dataset_id,
                            split_size=args.split_size,
                            num_splits=args.num_splits,
                            self_consist_id=args.self_consist_id,
                            num_runs=args.num_runs,
                            num_eval=args.num_eval,
                            models=args.models,
                            templates=args.templates,
                            extract_rule=args.extract_rule,
                            temperature=args.temperature,
                            num_workers=args.num_workers,
                            use_cache_samples=args.use_cache_samples,
                            use_cache_results=args.use_cache_results,
                            cache_dir=args.cache_dir,
                            max_new_tokens=args.max_new_tokens
                            )

  # Evaluate judges
  judges_evals.evaluate_judges()


if __name__ == "__main__":
  main()
