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
from tqdm import tqdm
if __name__ == '__main__':
    NUM_TEST_PROBLEMS = 1
    NUM_HYPOTHESIS = 32
    NUM_SUMMARY_HYPOTHESIS = 8
    NUM_PROGRAMS = 32

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',  default='configs/')
    parser.add_argument('--config_name',  default='arc')
    args = parser.parse_args()
    config = OmegaConf.load(f'{args.config_dir}{args.config_name}.yaml')

    problem_set = load_json(config['problem_set'])

    model = load_model_from_config(config)


    id_summary_hypothesis_programs = dict()
    raw_hypothesis_template, summary_hypothesis_template, python_implementation_template = load_templates()

    for key, problem in tqdm(list(problem_set.items())[:NUM_TEST_PROBLEMS]):
        id_summary_hypothesis_programs[key] = dict()
        id_summary_hypothesis_programs[key]['problem'] = problem
        id_summary_hypothesis_programs[key]['hypothesis_list'] = dict()

        raw_hypothesis_list = [eval(model(generate_hypothesis_prompt_fn(problem), raw_hypothesis_template))['Describing_how_to_transform_the_grid']
                               for _ in range(NUM_HYPOTHESIS)]  
        summary_hypothesis_list = eval(model(summarize_hypothesis_prompt_fn(raw_hypothesis_list, NUM_SUMMARY_HYPOTHESIS), summary_hypothesis_template))['rules']

        for i, summary_hypothesis in enumerate(summary_hypothesis_list):
            id_summary_hypothesis_programs[key]['hypothesis_list'][f'hypothesis_{i}'] = {}
            id_summary_hypothesis_programs[key]['hypothesis_list'][f'hypothesis_{i}']['hypothesis'] = summary_hypothesis
            python_implementation_list = [eval(model(implement_hypothesis_prompt_fn(problem, summary_hypothesis), python_implementation_template))['code']
                                          for _ in range(NUM_PROGRAMS)]
            id_summary_hypothesis_programs[key]['hypothesis_list'][f'hypothesis_{i}']['programs'] = python_implementation_list
    
    with open('logs/id_summary_hypothesis_programs_pricing_test.json', 'w+') as f:
        json.dump(id_summary_hypothesis_programs, f)
    
    with open('logs/id_summary_hypothesis_programs_pricing_test.json', 'r') as f:
        id_summary_hypothesis_programs_metrics = json.load(f)
    judge = ARCJudge()
    for problem_id, dataset in id_summary_hypothesis_programs_metrics.items():
        for hypothesis_id, hypothesis in dataset['hypothesis_list'].items():
            rolling_mean = []
            for program in hypothesis['programs']:
                for i in range(len(dataset['problem']['train'])):
                    result = judge.execute_function(program, np.array(dataset['problem']['train'][i]['input']))
                    rolling_mean.append(judge.percentage_correct(np.array(dataset['problem']['train'][i]['output']), result))
            dataset['hypothesis_list'][hypothesis_id]['mean'] = np.mean(rolling_mean)
    
    with open('logs/id_summary_hypothesis_programs_metrics_pricing_test.json', 'w+') as f:
        json.dump(id_summary_hypothesis_programs_metrics, f)
    