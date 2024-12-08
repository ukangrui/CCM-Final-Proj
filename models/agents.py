from typing import *
import openai
import requests
from pydantic import BaseModel
class ChatModel:
    def __init__(self):
        self.max_tokens = 4096
        self.temperature = 0.7

class GPTModel(ChatModel):
    def __init__(self, api_key, model_name='gpt-4o-mini', sys_prompt = ''):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.sys_prompt = sys_prompt
        self.client = openai.OpenAI(api_key=api_key)

    def get_response(self, usr_prompt, template = None, n = 1):
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": usr_prompt}
            ],
            response_format=template,
            max_tokens = self.max_tokens,
            n = n
        )
        return [response.choices[i].message.parsed for i in range(len(response.choices))]
    
    def __call__(self, *args):
        return self.get_response(*args)
    
class LLamaModel(ChatModel):
    def __init__(self, endpoint, model_name = "llama-3.1-70b", sys_prompt = '', n = 1):
        super().__init__()
        self.endpoint = endpoint
        self.model_name = model_name
        self.sys_prompt = sys_prompt
        self.n = n
    
    def get_response(self, usr_prompt, schema = None):
        json_input = {
        "messages": [
            {
            "content": self.sys_prompt,
            "role": "system",
            "name": "system"
            },
            {
            "content": usr_prompt,
            "role": "user",
            "name": "user"
            }

        ],
        "model": "llama-3.1-70b",
        "max_tokens": self.max_tokens,
        "temperature": self.temperature,
        "n":self.n,
        "guided_json": schema,
        
    }
        response = requests.post('https://vllm.ml1.ritsdev.top/v1/chat/completions', json=json_input).json()
        return response["choices"][0]["message"]["content"]

    def __call__(self, *args: Any):
        return self.get_response(*args)


