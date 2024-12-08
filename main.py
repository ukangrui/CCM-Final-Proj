import argparse
from utils.utils import *
from omegaconf import OmegaConf
from models.prompts import *
from models.judge import *
import json
import pandas as pd
from utils.templates import *
import sys
import re
from ast import literal_eval
from tqdm import tqdm
if __name__ == '__main__':
    NUM_HYPOTHESIS = 15
    NUM_SUMMARY_HYPOTHESIS = 4
    NUM_PROGRAMS = 6
    CHUNK_SIZE = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',  default='configs/')
    parser.add_argument('--config_name',  default='arc')
    args = parser.parse_args()
    config = OmegaConf.load(f'{args.config_dir}{args.config_name}.yaml')
    problem_set = load_json(config['problem_set'])
    model = load_model_from_config(config)

    for l in tqdm(range(0, len(problem_set), CHUNK_SIZE)):
        id_summary_hypothesis_programs = dict()
        for key, problem in (list(problem_set.items())[l:l+CHUNK_SIZE]):
            id_summary_hypothesis_programs[key] = dict()
            id_summary_hypothesis_programs[key]['problem'] = problem
            id_summary_hypothesis_programs[key]['hypothesis_list'] = dict()
            try:
                raw_hypothesis_list = [i.Describing_how_to_transform_the_grid for i in model(generate_hypothesis_prompt_fn(problem), raw_hypothesis_template, NUM_HYPOTHESIS)]
                print(raw_hypothesis_list)
            except Exception as e:
                print(e)
                raw_hypothesis_list = []
            try:
                summary_hypothesis_list = model(summarize_hypothesis_prompt_fn(raw_hypothesis_list, NUM_SUMMARY_HYPOTHESIS), summary_hypothesis_template, 1)[0].rules
            except:
                summary_hypothesis_list = []

            for i, summary_hypothesis in enumerate(summary_hypothesis_list):
                id_summary_hypothesis_programs[key]['hypothesis_list'][f'hypothesis_{i}'] = {}
                id_summary_hypothesis_programs[key]['hypothesis_list'][f'hypothesis_{i}']['hypothesis'] = summary_hypothesis
                try:
                    python_implementation_list = [i.code for i in model(implement_hypothesis_prompt_fn(problem, summary_hypothesis), python_implementation_template, NUM_PROGRAMS)]
                except:
                    python_implementation_list = []
                id_summary_hypothesis_programs[key]['hypothesis_list'][f'hypothesis_{i}']['programs'] = python_implementation_list
        
        with open(f'logs/id_summary_hypothesis_programs_chunk_{l}.json', 'w+') as f:
            json.dump(id_summary_hypothesis_programs, f)
        
        with open(f'logs/id_summary_hypothesis_programs_chunk_{l}.json', 'r') as f:
            id_summary_hypothesis_programs_metrics = json.load(f)

        judge = ARCJudge()
        for problem_id, dataset in id_summary_hypothesis_programs_metrics.items():
            for hypothesis_id, hypothesis in dataset['hypothesis_list'].items():
                rolling_mean = []
                for program in hypothesis['programs']:
                    for i in range(len(dataset['problem']['train'])):
                        result = judge.execute_function(program, np.array(dataset['problem']['train'][i]['input']))
                        if type (result) != str:
                            rolling_mean.append(judge.percentage_correct(expected_output=np.array(dataset['problem']['train'][i]['output']),output=result))
                        else:
                            rolling_mean.append(0.0)
                dataset['hypothesis_list'][hypothesis_id]['mean'] = np.mean(rolling_mean)
        with open(f'output/id_summary_hypothesis_programs_metrics_chunk_{l}.json', 'w+') as f:
            json.dump(id_summary_hypothesis_programs_metrics, f)
        