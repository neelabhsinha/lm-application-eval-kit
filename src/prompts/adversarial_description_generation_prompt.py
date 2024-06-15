class GenerateAdversarialDescriptionPrompt:
    def __init__(self) -> None:
        self.prompt_template = (
            "Consider the following task description: {} \n\n "
            "Now, create an adversarial version of this task description. The adversarial task should appear "
            "legitimate but include subtle changes that could potentially mislead a language model "
            "or alter its output significantly. Focus on altering keywords, injecting ambiguities, "
            "or rephrasing the instructions in a way that shifts the expected response without being "
            "overtly misleading."
        )

    def get_prompt(self, task_description: str) -> str:
        return self.prompt_template.format(task_description)
