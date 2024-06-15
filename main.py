import argparse
import os

from const import tasks, supported_metrics
from src.utils.eval import evaluate
from src.utils.analyze_overall_performances import ModelPerformanceAnalysisUtil
from src.dataset_analysis.super_natural_instructions_analyzer import SuperNaturalInstructionsAnalyzer
from src.utils.compute_metrics import compute_metrics


def configure_huggingface():
    try:
        hf_token = os.getenv('HF_API_KEY')  # Make sure to add HF_API_KEY to environment variables
        # Add it in .bashrc or .zshrc file to access it globally
        os.environ['HF_TOKEN'] = hf_token
    except (TypeError, KeyError):
        print('Not able to set HF token. Please set HF_API_KEY in environment variables.')


def get_args():
    parser = argparse.ArgumentParser(
        description='Project for evaluating different large language models (LLMs) on various tasks across multiple'
                    ' domains, categories, and types of reasoning.')

    # General execution parameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Define the batch size for model training or evaluation. Default is 1.")
    parser.add_argument('--force_recompute', action='store_true',
                        help='Force the re-computation of all evaluation metrics, even if they have been computed'
                             ' previously.')

    # Task-related parameters
    parser.add_argument("--task", type=str, default='eval', choices=tasks,
                        help="Specify the task to perform. Options are based on predefined tasks in the 'tasks'"
                             " module.")
    parser.add_argument("--model_name", nargs='+', default=None,
                        help="List of model names to be used for the evaluation or any other specified task.")
    parser.add_argument("--split", type=str, default='test', choices=['train', 'test'],
                        help="Define the data split to be used. Options are 'train' or 'test'. Default is 'test'.")

    # Prompt customization (definitions, in-context examples)
    parser.add_argument('--icl_examples', type=int, default=0,
                        help='Specify the number of in-context learning (ICL) examples to use. Default is 0.')
    parser.add_argument('--add_definition', action='store_true',
                        help='Enable this option to include definitions in the prompts used for tasks.')
    parser.add_argument('--add_paraphrased_definition', action='store_true',
                        help='Enable this option to include paraphrased definitions in the prompts, enhancing task'
                             ' clarity.')
    parser.add_argument('--add_adversarial_definition', action='store_true',
                        help='Include adversarial definitions in the prompts to test model robustness.')
    parser.add_argument('--add_explanation', action='store_true',
                        help='Include explanations for in-context learning examples to provide more context.')

    # Sampling configuration (if none is selected, decoder defaults to greedy sampling)
    parser.add_argument('--do_sample', action='store_true',
                        help='Activate sampling mode during model output generation to introduce variability.')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Limit the sample to the top k most likely tokens. Used to control the randomness of'
                             ' output predictions.')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Set the cumulative probability cutoff for sampling. Tokens with cumulative probabilities'
                             ' up to this threshold will be considered.')

    # Filtering and selection options (filtering entities are taken from const.py file)
    parser.add_argument('--instance_per_task', type=int, default=50000,
                        help='Set the maximum number of instances per task to process. Default is 50000.')
    parser.add_argument('--filter_domains', action='store_true',
                        help='Enable this option to apply a domain-specific filter during result aggregation, using'
                             ' predefined lists in constants.')
    parser.add_argument('--filter_categories', action='store_true',
                        help='Apply category-specific filters during result aggregation, based on predefined'
                             ' lists in constants.')
    parser.add_argument('--filter_reasoning', action='store_true',
                        help='Activate this to apply reasoning-specific filters during the results aggregation process,'
                             ' according to predefined lists.')

    # Evaluation configuration
    parser.add_argument('--metric', default='bert_score_recall', type=str, choices=supported_metrics,
                        help='Specify the evaluation metric to use. Default is "rougeL". Options might include'
                             ' various NLP-specific metrics like BLEU, METEOR, etc., depending on what is implemented.')

    # Checkpoint handling (if evaluating from a local checkpoint)
    parser.add_argument('--checkpoint', type=str, default='none',
                        help='Specify the checkpoint folder name if resuming from a saved state. Use "none"'
                             ' to start from scratch.')

    return parser.parse_args()


if __name__ == '__main__':
    configure_huggingface()
    args = get_args()
    ckp = None if args.checkpoint == 'none' else args.checkpoint
    if args.task == 'analyze_dataset':
        analyzer = SuperNaturalInstructionsAnalyzer(split=args.split, instance_per_task=args.instance_per_task)
        analyzer.save_analysis_results()
    if args.task == 'eval':
        evaluate(args.model_name[0], args.batch_size, args.icl_examples, add_definition=args.add_definition,
                 add_explanation=args.add_explanation, add_paraphrased_definition=args.add_paraphrased_definition,
                 add_adversarial_definition=args.add_adversarial_definition, filter_categories=args.filter_categories, 
                 filter_domains=args.filter_domains, filter_reasoning=args.filter_reasoning,
                 instance_per_task=args.instance_per_task, do_sample=args.do_sample, top_k=args.top_k, top_p=args.top_p,
                 checkpoint=ckp)
    if args.task == 'compute_metrics':
        compute_metrics(args.metric, args.force_recompute)
    if args.task == 'collect_results':
        obj = ModelPerformanceAnalysisUtil(metric=args.metric, filter_categories=args.filter_categories,
                                           filter_domains=args.filter_domains, filter_reasoning=args.filter_reasoning)
        obj.get_results(models=args.model_name)
        print('Results collected successfully.')
