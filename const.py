project_root = '.'  # project root directory
project_name = 'open-lm-evaluation-framework'  # project name

# ------ Execution-specific constants ------

# Dataset location (change the directory path here to the dataset)
source_dataset_dir = '/nethome/nsinha68/flash/datasets/super-natural-instructions/source_data'
have_paraphrased_definitions = True  # whether to include paraphrased definitions during data loading
have_adversarial_definitions = True  # whether to include adversarial definitions during data loading

# ------ End of execution-specific constants ------

# Directories (edit the source_dataset_dir as needed)
metadata_dir = f'{project_root}/metadata'  # metadata directory
cache_dir = f'{project_root}/cache'  # cache directory for HuggingFace
results_dir = f'{project_root}/results'  # directory to store model predictions and evaluation results
aggregated_results_dir = f'{project_root}/aggregated_results' # directory to store final tables, charts
dataset_analysis_dir = f'{project_root}/dataset_analysis' # directory to store dataset analysis results

tasks = ['train', 'eval', 'analyze_dataset', 'collect_results', 'compute_metrics', 'analyze_dataset']
# tasks to perform

# More readable Model Names (for results)
beautified_model_names = {
    'gemma-2b': 'Gemma-2B',
    'gemma-7b': 'Gemma-7B',
    'Mistral-7B-v0.3': 'Mistral-7B',
    'Meta-Llama-3-8B': 'Llama-3-8B',
    'falcon-11B': 'Falcon-2-11B',
    'gemma-2b-it': 'Gemma-2B-I',
    'Phi-3-mini-128k-instruct': 'Phi-3-mini-128k-I',
    'Mistral-7B-Instruct-v0.3': 'Mistral-7B-I',
    'gemma-7b-it': 'Gemma-7B-I',
    'Meta-Llama-3-8B-Instruct': 'Llama-3-8B-I',
    'gpt-3.5-turbo': 'GPT-3.5-T',
    'gpt-4o': 'GPT-4o',
    'deepseek-v2': 'DS-2'
}

# Filters for Accumulating results

# Domains to consider
domains_filter = [
    "Fiction", "Books",  # Art and Literature
    "Economics", "Law", "Government and Politics", "History",  # Social Sciences and Humanities
    "Computer Science", "Natural Science",  # Science and Technology
    "Nutrition", "Food",  # Health and Medical
    "Social Media", "News"  # Media and Entertainment
]

# Categories to consider
categories_filter = [
    "Data to Text", "Title Generation", "Question Rewriting",  # Generation
    "Word Analogy", "Grammar Error Correction",  # Linguistic Relationships
    "Coreference Resolution", "Dialogue Act Recognition", "Textual Entailment", "Overlap Extraction",
    # Semantic and Pragmatic Analysis
    "Keyword Tagging", "Answerability Classification", "Cause Effect Classification"  # Classification and Recognition
]

# Reasoning types to consider
reasoning_filter = [
    "Causal Reasoning", "Analogical Reasoning", "Commonsense Reasoning",  # Comparative and Relational Analysis
    "Deductive Reasoning", "Abductive Reasoning", "Logical Reasoning",  # Formal Logic
    "Multihop Reasoning", "Cross-document Reasoning",  # Complex Inference and Analysis
    "Quantitative Reasoning", "Temporal Reasoning"  # Specific Contextual Reasoning
]

distinctive_colors = [
    '#e6194b', '#3cb44b', '#ffc43a', '#4e4d6d', '#4363d8', '#c19d6d',
    '#911eb4', '#a64d79', '#614051', '#ea780c', '#000075', '#808000',
    '#008080', '#9a6324', '#800000', '#808080'
]
