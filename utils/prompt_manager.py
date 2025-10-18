import json
from pathlib import Path

class PromptManager:
    def __init__(self, prompt_file: str = 'prompts.json'):
        self.prompt_path = Path(__file__).parent.parent / prompt_file
        with open(self.prompt_path, 'r') as f:
            self.prompts = json.load(f)

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        prompt_template = self.prompts.get(prompt_name)
        if not prompt_template:
            raise ValueError(f"Prompt '{prompt_name}' not found in {self.prompt_path}")
        return prompt_template.format(**kwargs)
