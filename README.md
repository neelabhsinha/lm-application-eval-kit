# Evaluating Open Language Models Across Task Types, Application Domains, and Reasoning Types: An In-Depth Experimental Analysis

---
**Abstract** - The rapid rise of Language Models (LMs) has increased their use in many applications. However, state-of-the-art (SOTA) LLMs cannot always be used due to constraints like model size, associated cost or proprietary restrictions. With up and coming open, smaller-scale LMs, many more applications can utilize their capabilities. However, a single LM may not be globally best for all use cases. This work conducts an in-depth experimental analysis of the performance of 10 smaller, open LMs in terms of semantic correctness of outputs across three aspects: task types, application domains and reasoning types, using diverse prompt styles. We show that under different scenarios, different LMs give best results, and provide a comparative analysis of LMs and prompt styles using a proposed three-tier schema of aspects for their appropriate selection depending on use-case and other constraints. We also show that if selected and used properly, these LMs can compete with, and sometimes outperform, SOTA LLMs like DeepSeek-v2, GPT-3.5-Turbo, and GPT-4o.

---
## Setup

Clone the repository using the following command on your CLI -
    
    git clone https://github.com/neelabhsinha/open-lm-evaluation-framework.git
    cd open-lm-evaluation-framework

Install the required packages using the following command -
    
        pip install -r requirements.txt

Add your HuggingFace API key to the environment variables as `HF_API_KEY` to download the models from the HuggingFace model hub.

    export HF_API_KEY='your_huggingface_api_key_here'

[OPTIONAL] If you want to use GPT for generating paraphrases or adversarial task definitions, or for evaluation, add your OpenAI API key to the environment variables as `OPENAI_API_KEY`.

    export OPENAI_API_KEY='your_openai_api_key_here'

Download the dataset from the (https://instructions.apps.allenai.org/) website.

Go to `const.py` and set the following constants -
- `source_dataset_dir` - path to the downloaded dataset
- [OPTIONAL] `domains_filter`, `categories_filter`, `reasoning_filter` - add the domains, task types or reasoning types here if trying to experiment on a customized subset.
- [OPTIONAL] `beautified_model_names` - If you want to beautify the model names in the results, add the mapping here. Note, the key should be after the first slash(/) in the HuggingFace model key. For example, for google/gemma-2b-it, the key should be gemma-2b-it.

## Execution

main.py is the entry point of the code.

For evaluating an LM, make a decision on the following parameters -
- LM name - get the name of the LM as per the HuggingFace model key (for example, google/gemma-2b-it for Gemma-2B-Instruction tuned)
- instances per task - number of instances to evaluate per task definition
- batch size - number of instances to evaluate in a batch
- prompt style - prompt style to use for evaluation (definition and examples)
- Sampling techniques (if needed)

### Detailed Command-Line Options and Project Structure

The `main.py` script supports various command-line options to configure the execution of tasks such as evaluating models, analyzing datasets, and generating reports. Below are the available options and their descriptions:

#### General Options

- `--batch_size INT`: Defines the batch size for model training or evaluation processes. The default value is `1`. This parameter helps in managing memory usage and performance during model operations.

- `--task STRING`: Selects the task to perform. Available options include `eval` for model evaluations, `analyze_dataset` for dataset analysis, `compute_metrics` for metrics computation, and `collect_results` for aggregating and analyzing results. Each task triggers different components of the script.

- `--model_name MODEL1 MODEL2 ...`: Specifies one or more model names to be used for the evaluation. Models should be provided as a space-separated list. This allows flexibility in testing multiple models in a single run.

- `--split {train,test}`: Determines the data split to use during tasks. Options are `train` for training data and `test` for test data, with `test` being the default.

#### Prompt Style Options

- `--icl_examples INT`: Sets the number of in-context learning examples to include in the evaluation prompts. The default is `0`, meaning no in-context examples are used unless specified.

- `--add_definition`: Includes definitions in the prompts to provide clearer task instructions to the model.

- `--add_paraphrased_definition`: Adds paraphrased versions of the task definitions in the prompts, which can help in evaluating model understanding and flexibility.

- `--add_adversarial_definition`: Incorporates adversarial definitions into the prompts to test the robustness of the models against potentially confusing or misleading information.

- `--add_explanation`: Includes explanations along with in-context learning examples to give additional context that may aid the model in generating more accurate responses.

Note, only one of the `--add_definition`, `--add_paraphrased_definition`, and `--add_adversarial_definition` options can be used at a time.

#### Sampling Configuration

- `--do_sample`: Activates sampling mode during model output generation. This option introduces variability in the outputs by not always choosing the most likely next token.

- `--top_k INT`: Limits the sample to the top `k` most likely tokens when generating model outputs. This parameter controls the diversity of the responses by narrowing the token choice to a set number.

- `--top_p FLOAT`: Sets a threshold for nucleus sampling, where the cumulative probability of considered tokens must not exceed this value. It helps in controlling the randomness of the generated outputs by focusing on a subset of likely tokens.

#### Filtering and Selection Options

Use these options to filter and execute a task only on a small subset of entities from each aspect during the evaluation and analysis processes. The filter has to be assigned in `const.py`

- `--filter_domains`: Applies a domain-specific filter during result aggregation, using predefined lists. This allows focusing the analysis on specific areas of interest.

- `--filter_categories`: Enables filtering the results based on predefined categories, which helps in segmenting the analysis according to different types of tasks or content.

- `--filter_reasoning`: Activates reasoning-specific filters during results aggregation, focusing on how different models handle various reasoning tasks.

#### Evaluation and Metrics Configuration

- `--metric STRING`: Specifies the metric to use for evaluation. Default is `bert_score_recall`, but other metrics like BLEU, ROUGE, and METEOR can also be specified depending on the implementation.

- `--force_recompute`: Forces the re-computation of metrics, even if they have been previously calculated. This is useful when changes to the evaluation protocol or model configurations need to be reflected in new results.

- `--checkpoint PATH`: Points to a checkpoint directory for resuming an evaluation from a previously saved state. If set to `none`, the evaluation starts from scratch.

These options provide extensive control over how models are evaluated and analyzed, allowing users to tailor the process to their specific needs.
 
#### Project Structure

The project structure is organized into several directories and files that help manage the codebase and resources effectively. Below is an overview of the key components (not all will be available first, some may get created during execution as needed):
- `aggregated_results`: Contains the aggregated results from model evaluations and analyses, stored in CSV format or as visualizations.
- `cache`: Stores Huggingface cache
- `dataset_analysis`: Outputs of dataset analysis.
- `metadata`: Contains metadata files for dataset, including paraphrased and adversarial prompts if generated.
- `results`: Stores the raw results from model evaluations.
- `src`: Contains the source code for the project, including scripts for model evaluation, dataset analysis, and result aggregation.
- `const.py`: Defines constants and configurations used throughout the project, such as file paths, filters, and prompt styles.
- `main.py`: The main script for executing tasks and managing the evaluation process.
- `requirements.txt`: Lists the required Python packages for the project, which can be installed using `pip install -r requirements.txt`.
- `README.md`: Provides an overview of the project, setup instructions, and usage guidelines.

The `src` directory contains the following subdirectories:
- `dataset_analysis/su`

#### Example Command to analyze a particular split of the dataset -
    
        python main.py --task analyze_dataset --split test

#### Example Command to evaluate an LM -
    
        python main.py --task eval --model_name google/gemma-2b-it --instances_per_task 100 --batch_size 4 --add_definition --icl_examples 4 --do_sample --top_k 10

#### Example Command to compute metrics for a particular model -
        
            python main.py --task compute_metrics --metric bert_score_recall --model_name google/gemma-2b-it

#### Example Command to collect visualizations and results for a particular model -
        
            python main.py --task collect_results --metric bert_score_recall

## Citation

If you use this codebase or the dataset in your research, please cite the following paper -

```
CITATION COMING SOON
```




