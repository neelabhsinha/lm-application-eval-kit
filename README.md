# Evaluating Open Language Models Across Task Types, Application Domains, and Reasoning Types: An In-Depth Experimental Analysis

---
**Abstract** - The rapid rise of Language Models (LMs) has expanded their use in several applications. Yet, due to constraints of model size, associated cost or proprietary restrictions, utilizing state-of-the-art (SOTA) LLMs is not always feasible. With open, smaller LMs emerging, more applications can leverage their capabilities, but selecting the right LM can be challenging. This work conducts an in-depth experimental analysis of the semantic correctness of outputs of 10 smaller, open LMs across three aspects: *task types*, *application domains* and *reasoning types*, using diverse prompt styles. We demonstrate that most effective models and prompt styles vary depending on the specific requirements. Our analysis provides a comparative assessment of LMs and prompt styles using a proposed three-tier schema of aspects for their strategic selection based on use-case and other constraints. We also show that if utilized appropriately, these LMs can compete with, and sometimes outperform, SOTA LLMs like DeepSeek-v2, GPT-3.5-Turbo, and GPT-4o.

---
## Setup

STEP 1: [REQUIRED] Clone the repository.

STEP 2: [REQUIRED] Install the required packages using the following command -

```commandline
pip install -r requirements.txt
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

To use the `--add_paraphrased_definition` or `--add_adversarial_definition` options, you should need Step 4 of Setup to be completed. The definitions will be generated using GPT-3.5-Turbo and only the first time. They will be added to `metadata/` directory to future use.

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

## Our Results

Overall Performance of LMs on different metrics are given below -

| Model            | BLEU  | ROUGE-1    | ROUGE-2   | ROUGE-L    | METEOR    | BERT Score Precision | BERT Score Recall | BERT Score F1 | Best Instruction                |
|------------------|-------|------------|-----------|------------|-----------|----------------------|-------------------|---------------|---------------------------------|
| Gemma-2B         | 6.382 | 22.036     | 7.883     | 21.235     | 18.120    | 78.218               | 86.410            | 81.881        | 4 examples with definition      |
| Mistral-7B       | 0.290 | 1.174      | 0.542     | 1.085      | 1.993     | 49.248               | 58.408            | 53.395        | 8 examples with definition      |
| Gemma-7B         | 4.422 | 18.175     | 5.887     | 17.493     | 16.137    | 71.858               | 81.055            | 75.942        | 0 examples with definition      |
| Llama-3-8B       | 3.706 | 16.379     | 5.348     | 15.302     | 14.957    | 75.520               | 82.727            | 78.804        | 0 examples with definition      |
| Falcon-2-11B     | 4.285 | 16.885     | 6.461     | 16.013     | 16.450    | 79.652               | 86.184            | 82.718        | 8 examples with definition      |
| Gemma-2B-I       | 5.756 | 27.564     | 8.084     | 26.236     | 20.620    | 84.555               | 88.058            | 86.189        | 2 examples with definition      |
| Phi-3-mini-128k-I| 1.279 | 7.169      | 3.173     | 6.265      | 8.046     | 55.470               | 60.282            | 57.720        | 0 examples without definition   |
| Mistral-7B-I     | 14.069| 51.957     | 14.665    | 50.119     | 35.551    | 91.289               | 93.755            | 92.394        | 8 examples with definition      |
| Gemma-7B-I       | 0.972 | 8.642      | 3.229     | 7.964      | 12.568    | 78.184               | 85.142            | 81.483        | 0 examples with definition      |
| Llama-3-8B-I     | 0.953 | 4.682      | 2.191     | 4.233      | 8.312     | 74.231               | 84.332            | 78.888        | 8 examples without definition   |

**NOTE - You can refer all visualizations and tables of the paper in the `paper_results` directory.**

## Citation

If you use this codebase or our analysis in your research, please cite our paper.




