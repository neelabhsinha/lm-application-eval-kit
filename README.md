# Are Small Language Models Ready to Compete with Large Language Models for Practical Applications?

![Type](https://img.shields.io/badge/arXiv-2406.11402-yellow)
![Concepts](https://img.shields.io/badge/Concepts-Deep_Learning,_Natural_Language_Processing-blue)
![Language](https://img.shields.io/badge/Language-Python-red)
![Libraries](https://img.shields.io/badge/Libraries-PyTorch,_HuggingFace-green)


## Overview

Original Implementation of the paper Are Small Language Models Ready to Compete with Large Language Models for Practical Applications?, accepted at Accepted at The Fifth Workshop on Trustworthy Natural Language Processing (TrustNLP 2025) in Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), 2025.

arXiv Link: [2406.11402](https://arxiv.org/abs/2406.11402)

ACL Anthology Link: [2025.trustnlp-main.25](https://aclanthology.org/2025.trustnlp-main.25/)


## Setup

STEP 1: [REQUIRED] Clone the repository.

STEP 2: [REQUIRED] Install the required packages using the following command -

```commandline
pip install -r requirements.txt
```

Flash-attention-2 has to be installed separately using -
```commandline
pip install flash-attn --no-build-isolation
```

STEP 3: [REQUIRED] Add your HuggingFace API key to the environment variables (or bashrc/zshrc) as `HF_API_KEY` to download the models from the HuggingFace model hub.

```
export HF_API_KEY='your_huggingface_api_key_here'
```
If you have your API key with some other name, you can change the name in the `main.py` file.

STEP 4: [OPTIONAL] If you want to use GPT for generating paraphrases or adversarial task definitions, or for evaluation, add your OpenAI API key to the environment variables (or bashrc/zshrc) as `OPENAI_API_KEY`.

```
 export OPENAI_API_KEY='your_openai_api_key_here'
```
If you have your API key with some other name, you can change the name in the `src/models/gpt.py` file.

Additionally, go  to `const.py` and set the following constants -
- [REQUIRED] `have_paraphrased_definitions` - set to `True` if you want to generate paraphrased definitions for the tasks.
- [REQUIRED] `have_adversarial_definitions` - set to `True` if you want to generate adversarial definitions for the tasks.

If you don't want to generate or use paraphrased definitions, or use GPT for evaluation, you can skip the above step.

STEP 5: [OPTIONAL] If you want to evaluate DeepSeek-v2, add your DeepSeek API key to the environment variables (or bashrc/zshrc) as `DEEPSEEK_API_KEY`.

```
export DEEPSEEK_API_KEY='your_deepseek_api_key_here'
```
If you have your API key with some other name, you can change the name in the `src/models/deepseek.py` file.

STEP 6: [REQUIRED] Download the dataset from the (https://instructions.apps.allenai.org/) website.

Go to `const.py` and set the following constants -
- [REQUIRED] `source_dataset_dir` - path to the downloaded dataset
- [OPTIONAL] `domains_filter`, `categories_filter`, `reasoning_filter` - add the domains, task types or reasoning types here if trying to experiment on a customized subset.
- [OPTIONAL] `beautified_model_names` - If you want to beautify the model names in the results, add the mapping here. Note, the key should be after the first slash(/) in the HuggingFace model key. For example, for google/gemma-2b-it, the key should be gemma-2b-it.

## Execution

main.py is the entry point of the code.

For evaluating an LM, make a decision on the following parameters -
- LM name - get the name of the LM as per the HuggingFace model key (for example, google/gemma-2b-it for Gemma-2B-Instruction tuned)
- instance per task - number of instances to evaluate per task definition
- batch size - number of instances to evaluate in a batch
- prompt style - prompt style to use for evaluation (definition and examples)
- Sampling techniques (if needed)

### Execution Flow -

General steps to follow are -
1. Run evaluation for models. The predictions will be stored in the `results` directory. Two files - `predictions.csv` having predictions and metrics for each task instance, and `results_statistics.csv` describing the statistics related to the prediction.
2. Compute metrics for the results. The same two files as above will be updated with specified supported metric for all the files in `results` directory.
3. Collect results - Generate all statistics summary and visualizations. Radar charts and line graphs like the paper will be generated for elements in the filter elements (Check Step 6 above) using the model name specified in `beautified_model_names` if present.

Note -
- The Radar chart results will always be generated for the best prompt style for that entity
- The first time you run the code, a dataset metadata will be created and stored in `metadata/` directory. So, first execution will take time. But next time onwards, the existing metadata will be read, unless deleted.
- For result collection, by default, all models in `results` directory will be considered. If you want to consider only a subset of models, you can specify the models in the `--model_name` parameter by space separating the model names (use the `beautified_model_name` values here, not HuggingFace default).
- You can also change filters. We also support collecting results on a different metric. Refer to the command-line options below.

#### Example Command to analyze a particular split of the dataset -
```commandline
python main.py --task analyze_dataset --split test --instance_per_task 100
```
#### Example Command to evaluate an LM -
 
```commandline
python main.py --task eval --model_name google/gemma-2b-it --instance_per_task 100 --batch_size 4 --add_definition --icl_examples 4 --do_sample --top_k 10
```
Options can be changed based on the need. Adjust batch size based on your available GPU memory. Always specify the model name as per the HuggingFace model key. More on sampling techniques and prompt styles can be found in the detailed command-line options guide below.

#### Example Command to compute metrics for a particular model -
        
```commandline
python main.py --task compute_metrics --metric bert_score_recall --force_recompute
```
Force recompute is optional. Setting it to true will recompute the metrics even if they are already computed. If it's false, available ones will be skipped.
Supported metrics can be found in the detailed command-line options guide below.

#### Example Command to collect visualizations and results for a particular model -

```commandline
python main.py --task collect_results --metric bert_score_recall
```
If no `model_name` is specified, all models in the `results` directory will be considered. If you want to consider only a subset of models, you can specify the models in the `--model_name` parameter by space separating the model names (use the `beautified_model_name` values here, not HuggingFace default).

For example, if the models are not added to `beautified_model_names` in `const.py`, you can use the HuggingFace model name after the first slash(/) as `model_name`.
```commandline
python main.py --task collect_results --metric bert_score_recall --model_name google/gemma-2b-it google/falcon-2-11b
```
If they are added, you can use the beautified model names as -

```commandline
python main.py --task collect_results --metric bert_score_recall --model_name Gemma-2B Falcon-2-11B
```
Use a different metric if needed. See example below -

```commandline
python main.py --task collect_results --metric bleu --model_name Gemma-2B Falcon-2-11B
```    
                    

### Detailed Command-Line Options and Project Structure

The `main.py` script supports various command-line options to configure the execution of tasks such as evaluating models, analyzing datasets, and generating reports. Below are the available options and their descriptions:

#### General Options

- `--batch_size INT`: Defines the batch size for model training or evaluation processes. The default value is `1`. This parameter helps in managing memory usage and performance during model operations.

- `--task STRING`: Selects the task to perform. Available options include `eval` for model evaluations, `analyze_dataset` for dataset analysis, `compute_metrics` for metrics computation, and `collect_results` for aggregating and analyzing results. Each task triggers different components of the script.

- `--model_name MODEL1 MODEL2 ...`: Specifies one or more model names to be used for the evaluation. Models should be provided as a space-separated list. This allows flexibility in testing multiple models in a single run. When evaluating, always pass a single model name. If multiple is passed, first one will be considered. Multiple options is only for collecting results.

- `--split {train,test}`: Determines the data split to use during tasks. Options are `train` for training data and `test` for test data, with `test` being the default.

For evaluations, model_name will take HuggingFace model keys. For collecting results, model_name will take the keys specified in `beautified_model_names` in `const.py`, or the HuggingFace model name after the first slash(/).

Split is only used for data analysis.

#### Prompt Style Options

- `--icl_examples INT`: Sets the number of in-context learning examples to include in the evaluation prompts. The default is `0`, meaning no in-context examples are used unless specified.

- `--add_definition`: Includes definitions in the prompts to provide clearer task instructions to the model.

- `--add_paraphrased_definition`: Adds paraphrased versions of the task definitions in the prompts, which can help in evaluating model understanding and flexibility.

- `--add_adversarial_definition`: Incorporates adversarial definitions into the prompts to test the robustness of the models against potentially confusing or misleading information.

- `--add_explanation`: Includes explanations along with in-context learning examples to give additional context that may aid the model in generating more accurate responses.

To use the `--add_paraphrased_definition` or `--add_adversarial_definition` options, you should need Step 4 of Setup to be completed. The definitions will be generated using GPT-3.5-Turbo and only the first time. They will be added to `metadata/` directory to future use. Also, the flags of Step 4 needs to be `true` for evaluations if using adversarial or paraphrased definitions. Otherwise, it will just append a blank string to the task definition.

Note, only one of the `--add_definition`, `--add_paraphrased_definition`, and `--add_adversarial_definition` options can be used at a time.

#### Sampling Configuration (default is do_sample=False which does greedy decoding)

- `--do_sample`: Activates sampling mode during model output generation. This option introduces variability in the outputs by not always choosing the most likely next token.

- `--top_k INT`: Limits the sample to the top `k` most likely tokens when generating model outputs. This parameter controls the diversity of the responses by narrowing the token choice to a set number.

- `--top_p FLOAT`: Sets a threshold for nucleus sampling, where the cumulative probability of considered tokens must not exceed this value. It helps in controlling the randomness of the generated outputs by focusing on a subset of likely tokens.

#### Filtering and Selection Options

Use these options to filter and evaluate a model on a small subset of entities from each aspect during the evaluation and analysis processes. The filter has to be assigned in `const.py`. You can leave these empty to run on the entire dataset.
We didn't use these options when evaluating the model using our dataset in the main paper (we only created visualizations on this basis).

- `--filter_domains`: Applies a domain-specific filter during result aggregation, using predefined lists. This allows focusing the analysis on specific areas of interest.

- `--filter_categories`: Enables filtering the results based on predefined categories, which helps in segmenting the analysis according to different types of tasks or content.

- `--filter_reasoning`: Activates reasoning-specific filters during results aggregation, focusing on how different models handle various reasoning tasks.

#### Evaluation and Metrics Configuration

- `--metric STRING`: Specifies the metric to use for evaluation. Default is `bert_score_recall`, but other metrics are also supported and include `bleu`, `rouge1`, `rouge2`, `rougeL`, `meteor`, `bert_score_f1`, `bert_score_precision`.

- `--force_recompute`: Forces the re-computation of metrics, even if they have been previously calculated. This is useful when changes to the evaluation protocol or model configurations need to be reflected in new results.

#### Evaluating a Local Checkpoint -

- `--checkpoint PATH`: Points to a checkpoint directory for evaluating an LM from a previously saved state. The state should be saved using `save_pretrained()` method of HuggingFace and it should be in cache directory. If set to `none`, the evaluation starts from the model checkpoint available on HuggingFace cloud.

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
- `dataset_analysis/super_natural_instructions_analyzer.py`: Analyzes the dataset and generates statistics and visualizations.
- `handler/exit_handler.py`: Handles the exit of the program.
- `loader/super_natural_instructions.py`: Perform dataset operations - load a task instance, paraphrased/adversarial definitions, etc.
- `loader/super_natural_instructions_loader.py`: Acts like a data loader, filtering the data, collecting it using previous file and batching
- `metrics` - Contains the metric computation scripts. Supported metrics for now are `BERTScore`, `bleu`, `rouge`, and `meteor`.
- `models` - Contains the model evaluation scripts. Supported models for now are `GPT`, `DeepSeek`, and all `HuggingFace` models.
- `prompts/super_natural_instructions_prompt.py` - Takes the dataset and generates prompt as per the provided prompt style.
- `prompts/adversarial_definition_prompt.py` - Generates prompt for getting adversarial task definitions for the tasks.
- `utils/analyze_overall_performances.py` - Generates CSVs and visualizations for the executions in `results` directory.
- `utils/compute_metrics.py` - Computes all metrics (or any specific one if needed) for the results. Updates the original `results` files with the computed metrics.
- `utils/gpu_stats.py` - Provides GPU stats for the system.
- `utils/radar_chart_generator.py` - Generates radar charts for the results.
- `utils/results_io_util.py` - Some read/write utilities for results.

## Citation

```bibtex
@inproceedings{sinha-etal-2025-small,
    title = "Are Small Language Models Ready to Compete with Large Language Models for Practical Applications?",
    author = "Sinha, Neelabh  and
      Jain, Vinija  and
      Chadha, Aman",
    editor = "Cao, Trista  and
      Das, Anubrata  and
      Kumarage, Tharindu  and
      Wan, Yixin  and
      Krishna, Satyapriya  and
      Mehrabi, Ninareh  and
      Dhamala, Jwala  and
      Ramakrishna, Anil  and
      Galystan, Aram  and
      Kumar, Anoop  and
      Gupta, Rahul  and
      Chang, Kai-Wei",
    booktitle = "Proceedings of the 5th Workshop on Trustworthy NLP (TrustNLP 2025)",
    month = may,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.trustnlp-main.25/",
    pages = "365--398",
    ISBN = "979-8-89176-233-6",
    abstract = "The rapid rise of Language Models (LMs) has expanded their use in several applications. Yet, due to constraints of model size, associated cost, or proprietary restrictions, utilizing state-of-the-art (SOTA) LLMs is not always feasible. With open, smaller LMs emerging, more applications can leverage their capabilities, but selecting the right LM can be challenging as smaller LMs don`t perform well universally. This work tries to bridge this gap by proposing a framework to experimentally evaluate small, open LMs in practical settings through measuring semantic correctness of outputs across three practical aspects: task types, application domains and reasoning types, using diverse prompt styles. It also conducts an in-depth comparison of 10 small, open LMs to identify best LM and prompt style depending on specific application requirement using the proposed framework. We also show that if selected appropriately, they can outperform SOTA LLMs like DeepSeek-v2, GPT-4o-mini, Gemini-1.5-Pro, and even compete with GPT-4o."
}
```




