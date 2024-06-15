import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from const import cache_dir
from src.models.deepseek import DeepSeekV2
from src.models.gpt import GPT

from collections import Counter


class LanguageModel:
    def __init__(self, model_name, for_eval=True):
        if 'local' in model_name:
            model_name.replace('/', '--')
        self.model_name = model_name
        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.for_eval = for_eval

    def get_model(self):
        if 'gpt' in self.model_name:
            model = GPT(model=self.model_name, mode='generate')
        elif self.model_name == 'deepseek-v2':
            model = DeepSeekV2()
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=cache_dir, device_map='auto',
                                                         trust_remote_code=True, quantization_config=self.nf4_config,
                                                         attn_implementation="flash_attention_2")
            value_counts = Counter(model.hf_device_map.values())
            total_values = sum(value_counts.values())
            value_percentages = {value: (count / total_values) * 100 for value, count in value_counts.items()}
            print('Distribution of weights across devices - ', value_percentages)
        print(f'Loaded model {self.model_name}')

        return model

    def get_tokenizer(self):
        if 'gpt' not in self.model_name and self.model_name != 'deepseek-v2':
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir, padding_side='left',
                                                      trust_remote_code=True)
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Padding token was not set, setting EOS token as padding token.")
        else:
            tokenizer = None
        return tokenizer
