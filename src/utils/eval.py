import pandas as pd
import sys
from tqdm import tqdm
import torch

from const import results_dir, cache_dir, categories_filter, domains_filter, reasoning_filter
from src.handler.exit_handler import EXIT
from src.loader.super_natural_instructions_loader import SuperNaturalInstructionsLoader
from src.metrics.bleu_score import BleuScore
from src.models.model import LanguageModel
from src.prompts.super_natural_instructions_prompt import SuperNaturalInstructionPrompt
from src.utils.results_io_util import write_results
from src.utils.gpu_stats import get_gpu_memory
from src.metrics.rouge import RougeScore


def evaluate(model_name, batch_size, examples, add_definition=True, add_paraphrased_definition=False,
             add_adversarial_definition=False, add_explanation=False, filter_categories=False,
             filter_domains=False, filter_reasoning=False, instance_per_task=50000,
             do_sample=False, top_k=None, top_p=None, checkpoint=None):
    categories = categories_filter if filter_categories else None
    domains = domains_filter if filter_domains else None
    reasonings = reasoning_filter if filter_reasoning else None
    parameters_dict = {'model_name': model_name, 'examples': examples, 'add_definition': add_definition,
                       'add_paraphrased_definition': add_paraphrased_definition,
                       'add_adversarial_definition': add_adversarial_definition,
                       'add_explanation': add_explanation, 'categories': categories, 'domains': domains,
                       'reasonings': reasonings, 'instance_per_task': instance_per_task, 'do_sample': do_sample,
                       'top_k': top_k, 'top_p': top_p, 'checkpoint': checkpoint}
    print('Parameters -')
    print(str(parameters_dict) + '\n\n')
    data_loader = SuperNaturalInstructionsLoader(split='test', categories=categories, domains=domains,
                                                 reasonings=reasonings, instance_per_task=instance_per_task,
                                                 batch_size=batch_size)
    prompt_util = SuperNaturalInstructionPrompt(example=examples, add_definition=add_definition,
                                                add_paraphrased_definition=add_paraphrased_definition,
                                                add_adversarial_definition=add_adversarial_definition,
                                                add_explanation=add_explanation)
    model_builder = LanguageModel(f'{cache_dir}/{checkpoint}' if checkpoint is not None else model_name)
    model = model_builder.get_model()
    tokenizer = model_builder.get_tokenizer()
    if 'gpt' not in model_name and 'deepseek' not in model_name:
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
    if 'gpt' in model_name:
        model_name = 'openai/' + model_name
    elif 'deepseek' in model_name:
        model_name = 'deepseek-ai/deepseek-v2'
    name = checkpoint if checkpoint is not None else ('pretrained--' + model_name.replace('/', '--'))
    name += (f'--positive_examples-{examples}--add_definition-{add_definition}--add_explanation-{add_explanation}'
             f'--add_paraphrased_definition-{add_paraphrased_definition}--add_adversarial_definition-'
             f'{add_adversarial_definition}'
             f'--do_sample-{do_sample}--top_k-{top_k}--top_p-{top_p}')
    results_path = f'{results_dir}/{name}'
    results_df = evaluation_loop(data_loader, prompt_util, tokenizer, model, do_sample, top_k, top_p, model_name,
                                 'Super Natural Instructions')
    write_results(results_df, results_path, parameters_dict)


def evaluation_loop(data_loader, prompt_util, tokenizer, model, do_sample, top_k, top_p, model_name, dataset_name):
    bleu_score = BleuScore()
    rouge_score = RougeScore()
    if 'gpt' not in model_name and 'deepseek' not in model_name:
        model.generation_config.pad_token_ids = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    results = {'prompt': [], 'candidate': [], 'reference': [],
               'domains': [], 'categories': [], 'reasoning': [],
               'input_language': [], 'output_language': [], 'instruction_language': [],
               'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': []}
    pbar = tqdm(data_loader, total=len(data_loader),
                desc=f'Evaluating Model {model_name} on {dataset_name}')
    count = 0
    for batch in pbar:
        try:
            prompts, label = prompt_util.get_prompt(batch)
            if 'gpt' not in model_name and 'deepseek' not in model_name:
                device = next(model.parameters()).device
                inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device=device)
                inputs = {key: value.to(dtype=torch.int32) for key, value in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=512)
                candidate_batch = tokenizer.batch_decode(outputs, do_sample=do_sample, top_k=top_k, top_p=top_p,
                                                         skip_special_tokens=True)
            else:
                candidate_batch = [model(prompt) for prompt in prompts]
            for i, (cand, inp) in enumerate(zip(candidate_batch, prompts)):
                if inp in cand:
                    cand = cand.replace(inp, '')
                if cand.startswith('\n'):
                    cand = cand[1:]
                candidate_batch[i] = cand
            reference_batch = label
            bleu = bleu_score.get_score(candidate_batch, reference_batch)
            rouge = rouge_score.get_score(candidate_batch, reference_batch)
            for candidate, ref, prompt, instance, rouge1, rouge2, rougeL, bl in zip(candidate_batch, reference_batch,
                                                                                    prompts, batch, rouge['rouge1'],
                                                                                    rouge['rouge2'], rouge['rougeL'],
                                                                                    bleu):
                results['prompt'].append(prompt)
                results['candidate'].append(candidate)
                results['reference'].append(ref)
                results['domains'].append(', '.join(instance['domains']))
                results['categories'].append(', '.join(instance['categories']))
                results['reasoning'].append(', '.join(instance['reasoning']))
                results['input_language'].append(', '.join(instance['input_language']))
                results['output_language'].append(', '.join(instance['output_language']))
                results['instruction_language'].append(', '.join(instance['instruction_language']))
                results['bleu'].append(bl * 100)
                results['rouge1'].append(rouge1 * 100)
                results['rouge2'].append(rouge2 * 100)
                results['rougeL'].append(rougeL * 100)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print('\nCuda Out of Memory Error: Clearing Cache', file=sys.stderr)
        if EXIT.is_set():
            return
        if count % 10 == 0:
            pbar.set_postfix(get_gpu_memory())
        count += 1
    results_df = pd.DataFrame(results)
    return results_df
