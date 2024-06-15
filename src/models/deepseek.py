from openai import OpenAI
import os


class DeepSeekV2:
    def __init__(self):
        api_key = os.getenv('DEEPSEEK_API_KEY')
        self.client = OpenAI(api_key=api_key, base_url='https://api.deepseek.com')
        self.model = 'deepseek-chat'
        self.instruction = 'You are an AI assistant designed to help users in completion of a task.'

    def __call__(self, text):
        messages = [
            {'role': 'system', 'content': self.instruction},
            {'role': 'user', 'content': text}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            output = response.choices[0].message.content
        except:
            output = 'Deepseek threw error with that input.'
        return output
