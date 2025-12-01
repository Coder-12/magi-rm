import os
from openai import OpenAI

class LLMInterface:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model_name
        self.temperature = temperature

    def chat(self, prompt: str, n: int = 1, max_tokens: int = 512):
        """Call the modern OpenAI Chat Completions API (2024+ syntax)."""
        responses = []
        for _ in range(n):
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            responses.append(completion.choices[0].message.content.strip())
        return responses