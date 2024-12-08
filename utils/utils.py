import json
import importlib
from models.agents import *
import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import *
from omegaconf import OmegaConf
from models.prompts import *
from datetime import datetime
from models.judge import *
import os


from pydantic import *
from typing import *
class raw_hypothesis_template(BaseModel):
    Describing_the_input_grid: str
    Describing_the_size_of_the_output_grid:str
    Describing_how_to_transform_the_grid:str

class summary_hypothesis_template(BaseModel):
    rules: List[str]

class python_implementation_template(BaseModel):
    code:str


def load_json(filename):
    with open(filename) as f:
        return json.load(f)

def save_json_response(response, save_dir, filename = None):
    if filename is None:
        start_idx = len([i for i in os.listdir(save_dir) if 'response' in i])
        filename = f'{save_dir}response_{start_idx}.json'
    with open(filename, 'w') as f:
        json.dump(response, f)
    return filename

def python_code_response_parser(solutions):
    outputs = []
    if not isinstance(solutions, list):
        solutions = [solutions]
    for solution in solutions:
        try:
            begin_delim = "```python"
            end_delim = "```"
            begin_idx = solution.index(begin_delim)
            end_idx = solution.index(end_delim, begin_idx+len(begin_delim))
            outputs.append(solution[begin_idx + len(begin_delim) : end_idx])
        except Exception as e:
            print(f"Error parsing solution: {e}")
            outputs.append("")
    return outputs 
    
def load_model_from_config(config):
    agent = config.agent
    if 'gpt' in agent.lower():
        api_key = config.get('api_key', '') 
        return GPTModel(api_key, sys_prompt=config.get('sys_prompt', ''))
    elif 'llama' in agent.lower():
        endpoint = config.get('endpoint', '')  
        return LLamaModel(endpoint, sys_prompt=config.get('sys_prompt', ''), n=config.get('n_completion', 1))
    else:
        raise ValueError(f"Unknown agent type: {agent}")
    
def load_judge_from_config(args):
    if 'usaco' in args.task.lower():
        return USACOJudge(args.log_dir)
    else:
        return ARCJudge()

async def response_wrapper(model, problem, solve_prompt_fn):
    usr_prompt = solve_prompt_fn(problem['description'])
    zero_response = await asyncio.to_thread(model, usr_prompt) 
    ### Enter your multi-agent multi-round logic here
    return zero_response

async def async_responses(model, problem_set: dict, solve_prompt_fn):
    async_responses = []
    for key, problem in problem_set.items():
        async_task = asyncio.create_task(response_wrapper(model,problem, solve_prompt_fn))
        async_responses.append(async_task)
    return await tqdm_asyncio.gather(*async_responses)

def final_responses(model, problem_set, solve_prompt_fn, save_dir, save = True, filename = None):
    keys = list(problem_set.keys())
    responses = asyncio.run(async_responses(model=model, problem_set=problem_set, solve_prompt_fn=solve_prompt_fn))
    responses = [python_code_response_parser(response) for response in responses]
    responses = dict(zip(keys, responses))
    if save:
        filename = save_json_response(responses, save_dir, filename)
    else:
        filename = None
    return responses, filename


import requests
def llama3_structured(schema,usermsg,sysmsg,max_tokens = 1024, temperature = 0.5, n=1):
    json_input = {
        "messages": [
            {
            "content": sysmsg,
            "role": "system",
            "name": "system"
            },
            {
            "content": usermsg,
            "role": "user",
            "name": "user"
            }

        ],
        "model": "llama-3.1-70b",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n":n,
        "guided_json": schema,
        
    }
    response = requests.post('https://vllm.ml1.ritsdev.top/v1/chat/completions', json=json_input).json()
    return response["choices"][0]["message"]["content"]
