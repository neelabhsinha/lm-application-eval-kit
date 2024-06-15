from openai import OpenAI
import os

from src.prompts.adversarial_definition_generation_prompt import GenerateAdversarialDescriptionPrompt


class GPT:
    def __init__(self, model='gpt-3.5-turbo', mode='paraphrase'):
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.mode = mode
        if mode == 'paraphrase':
            self.instruction = ('You are an AI assistant designed to paraphrase a definition of a task. You will be '
                                'provided with a paragraph that defines a particular task to be done. '
                                'Your task is to paraphrase the given definition so that it\'s interpretable by another '
                                'AI assistant to fulfill the task. Make sure to not omit any information from the '
                                ' paragraph. It might be necessary to complete the task. Only paraphrase it.')
        elif mode == 'generate':
            self.instruction = 'You are an AI assistant designed to help users in completion of a task.'
        elif mode == 'adversarial_definition':
            self.instruction = 'You are an AI assistant designed to create adversarial task descriptions of a task. '
            self.generate_adversarial_description_prompt = GenerateAdversarialDescriptionPrompt()

    def __call__(self, text):
        if self.mode == 'adversarial_definition':
            text = self.generate_adversarial_description_prompt.get_prompt(text)
        messages = [
            {'role': 'system', 'content': self.instruction},
            {'role': 'user', 'content': text}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        output = response.choices[0].message.content
        return output
