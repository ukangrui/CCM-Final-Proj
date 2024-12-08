import argparse
from utils.utils import *
from omegaconf import OmegaConf
from models.prompts import *
from models.judge import *
import json
import pandas as pd
from utils.templates import *
import sys
import asyncio
import aiohttp
import numpy as np

# Async function for making API calls
async def fetch_model_output(session, semaphore, prompt, template):
    async with semaphore:
        async with session.post('https://vllm.ml1.ritsdev.top/v1/chat/completions', json={
        "messages": [
            {
            "content": "You are a genius solving language puzzles.",
            "role": "system",
            "name": "system"
            },
            {
            "content": prompt,
            "role": "user",
            "name": "user"
            }

        ],
        "model": "llama-3.1-70b",
        "max_tokens": 2048,
        "temperature": 0.5,
        "n":1,
        "guided_json": template,
        
    }) as response:
            json_response = await response.json()  
            return json_response["choices"][0]["message"]["content"]
        
# Main function
async def main():
    NUM_TEST_PROBLEMS = 10
    NUM_HYPOTHESIS = 25
    NUM_SUMMARY_HYPOTHESIS = 5
    NUM_PROGRAMS = 25
    MAX_CONCURRENT_SESSIONS = 25

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default='configs/')
    parser.add_argument('--config_name', default='arc')
    args = parser.parse_args()
    config = OmegaConf.load(f'{args.config_dir}{args.config_name}.yaml')

    problem_set = load_json(config['problem_set'])
    gpt_model = load_model_from_config(config)

    aw_hypothesis_template_vllm, summary_hypothesis_template_vllm, python_implementation_template_vllm = load_templates()

    id_summary_hypothesis_programs = dict()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SESSIONS)

    async with aiohttp.ClientSession() as session:
        for key, problem in list(problem_set.items())[:NUM_TEST_PROBLEMS]:
            id_summary_hypothesis_programs[key] = dict()
            id_summary_hypothesis_programs[key]['problem'] = problem
            id_summary_hypothesis_programs[key]['hypothesis_list'] = dict()

            # Generate raw hypotheses concurrently
            raw_hypothesis_prompts = [generate_hypothesis_prompt_fn(problem) for _ in range(NUM_HYPOTHESIS)]
            raw_hypothesis_futures = [
                fetch_model_output(session, semaphore, prompt, aw_hypothesis_template_vllm) for prompt in raw_hypothesis_prompts
            ]
            raw_hypotheses = await asyncio.gather(*raw_hypothesis_futures)
            raw_hypothesis_list = [
                eval(raw_hypothesis)['Describing_how_to_transform_the_grid'] for raw_hypothesis in raw_hypotheses
            ]
            # Generate summary hypotheses
            summary_hypothesis_list = gpt_model(summarize_hypothesis_prompt_fn(raw_hypothesis_list, NUM_SUMMARY_HYPOTHESIS), summary_hypothesis_template).rules
            # Generate Python implementations for each summary hypothesis
            for i, summary_hypothesis in enumerate(summary_hypothesis_list):
                id_summary_hypothesis_programs[key]['hypothesis_list'][f'hypothesis_{i}'] = {}
                id_summary_hypothesis_programs[key]['hypothesis_list'][f'hypothesis_{i}']['hypothesis'] = summary_hypothesis

                implementation_prompts = [
                    implement_hypothesis_prompt_fn(problem, summary_hypothesis) for _ in range(NUM_PROGRAMS)
                ]
                implementation_futures = [
                    fetch_model_output(session, semaphore, prompt, python_implementation_template_vllm) for prompt in implementation_prompts
                ]
                python_implementation_responses = await asyncio.gather(*implementation_futures)
                python_implementation_list = []
                for index, implementation in enumerate(python_implementation_responses):
                    try:
                        python_implementation_list.append(eval(implementation)['code'])
                    except Exception as e:
                        print(f"Error parsing solution: {index}")
                        pass
                id_summary_hypothesis_programs[key]['hypothesis_list'][f'hypothesis_{i}']['programs'] = python_implementation_list

    # Save results
    with open('logs/id_summary_hypothesis_programs.json', 'w+') as f:
        json.dump(id_summary_hypothesis_programs, f)

    # Judge execution
    judge = ARCJudge()
    with open('logs/id_summary_hypothesis_programs.json', 'r') as f:
        id_summary_hypothesis_programs_metrics = json.load(f)

    for problem_id, dataset in id_summary_hypothesis_programs_metrics.items():
        for hypothesis_id, hypothesis in dataset['hypothesis_list'].items():
            rolling_mean = []
            for program in hypothesis['programs']:
                for i in range(len(dataset['problem']['train'])):
                    result = judge.execute_function(program, np.array(dataset['problem']['train'][i]['input']))
                    rolling_mean.append(judge.percentage_correct(np.array(dataset['problem']['train'][i]['output']), result))
            dataset['hypothesis_list'][hypothesis_id]['mean'] = np.mean(rolling_mean)

    with open('logs/id_summary_hypothesis_programs_metrics.json', 'w+') as f:
        json.dump(id_summary_hypothesis_programs_metrics, f)

# Run the async main function
if __name__ == '__main__':
    asyncio.run(main())
