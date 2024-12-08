import numpy as np
from typing import *
def get_colors():
    colors = [None] * 10
    colors[0] = {'color_name': 'black', 'rgb': '#000'}
    colors[1] = {'color_name': 'blue', 'rgb': '#0074D9'}
    colors[2] = {'color_name': 'red', 'rgb': '#FF4136'}
    colors[3] = {'color_name': 'green', 'rgb': '#2ECC40'}
    colors[4] = {'color_name': 'yellow', 'rgb': '#FFDC00'}
    colors[5] = {'color_name': 'grey', 'rgb': '#AAAAAA'}
    colors[6] = {'color_name': 'fuschia', 'rgb': '#F012BE'}
    colors[7] = {'color_name': 'orange', 'rgb': '#FF851B'}
    colors[8] = {'color_name': 'teal', 'rgb': '#7FDBFF'}
    colors[9] = {'color_name': 'brown', 'rgb': '#870C25'}
    return colors

def get_grid_str(grid):
    grid_str = str(np.array(grid))
    return grid_str

def generate_hypothesis_prompt_fn(task):
    COLORS = get_colors()
    prompt = """
You will be given a list of input-output pairs. Each input and output is a grid of numbers representing representing a visual grid. There is a SINGLE pattern that transforms each input grid to the corresponding output grid.'
The pattern may involve counting or sorting objects (e.g. sorting by size), comparing numbers (e.g. which shape or symbol appears the most? Which is the largest object? Which objects are the same size?), or repeating a pattern for a fixed number of time.
There are other concepts that may be relevant.
- Lines, rectangular shapes
- Symmetries rotations, translations.
- Shape upscaling or downscaling, elastic distortions.
- Containing / being contained / being inside or outside of a perimeter.
- Drawing lines, connecting points, orthogonal projections.
- Copying, repeating objects.
You should treat black cells as empty cells (backgrounds).
"""
    prompt += '\nThe number in the input grid can be mapped to the following colors:' + '; '.join([f"{c}:{COLORS[c]['color_name']}" for c in range(10)])
    prompt += '\nOutput the language description of the transformation.'
    prompt += 'Your description should be in the format:\nDescribing the input grid:{text}\n Describing the size of the output grid:{text}\n Describing how to transform the grid:{text}\n'
    prompt += 'Below are some examples of input-output pairs and description. After reading the examples, you will be provided with a new input-output pair marked in [question], and output the description of the transformation.'
    with open('./data/arc-prize-2024/example.txt', 'r') as f:
        prompt += f.read()
    prompt += '\n'
    prompt += '[question]\n'
    for i, t in enumerate(task['train']):
        prompt += f'Case {i}:\nInput:\n'
        prompt += get_grid_str(t['input']) + '\n'
        prompt += 'Output:\n'
        prompt += get_grid_str(t['output']) + '\n'
    prompt += '[question]\n'
    return prompt

def generate_hypothesis_prompt_finetuned_fn(task):
    COLORS = get_colors()
    prompt = """
You will be given a list of input-output pairs. Each input and output is a grid of numbers representing representing a visual grid. There is a SINGLE pattern that transforms each input grid to the corresponding output grid.'
The pattern may involve counting or sorting objects (e.g. sorting by size), comparing numbers (e.g. which shape or symbol appears the most? Which is the largest object? Which objects are the same size?), or repeating a pattern for a fixed number of time.
There are other concepts that may be relevant.
- Lines, rectangular shapes
- Symmetries rotations, translations.
- Shape upscaling or downscaling, elastic distortions.
- Containing / being contained / being inside or outside of a perimeter.
- Drawing lines, connecting points, orthogonal projections.
- Copying, repeating objects.
You should treat black cells as empty cells (backgrounds).
"""
    prompt += '\nThe number in the input grid can be mapped to the following colors:' + '; '.join([f"{c}:{COLORS[c]['color_name']}" for c in range(10)])
    prompt += 'Propose a hypothesis for solving this problem effectively:{text}\n'
    prompt += '[question]\n'
    for i, t in enumerate(task['train']):
        prompt += f'Case {i}:\nInput:\n'
        prompt += get_grid_str(t['input']) + '\n'
        prompt += 'Output:\n'
        prompt += get_grid_str(t['output']) + '\n'
    prompt += '[question]\n'
    return prompt


### Returns list of summarized hypothesis
def summarize_hypothesis_prompt_fn(hypothesis: List[str], num_hypothesis = 2) -> List[str]:
    prompt=  f"""
Given a list of rules, categorize them into {num_hypothesis} distinct categories based on their similarities. For each category, synthesize the rules into a single, specific rule that combines the ideas of all rules in that category, while clearly differentiating it from the other categories.
The new rule should be as specific as possible, following the format of the given rules.
The new rule should be applicable without any information from the original rules - i.e. it should be standalone.
Rules:
"""
    for i, h in enumerate(hypothesis):
        prompt += f'Rule {i}:\n'
        prompt += h + '\n'
    return prompt

### Returns list of list of implemented hypothesis
def implement_hypothesis_prompt_fn(task, summarized_hypothesis: str) -> str:
    prompt = """
'You will be given a list of input-output pairs. Each input and output is a grid of numbers. There is a single pattern that transforms each input grid to the corresponding output grid. You will be given a new input grid, and your job is to infer its corresponding output grid with the same transformation. The example input-output pairs are given below:'
"""
    for i, t in enumerate(task['train']):
        prompt += f'Example {i}:\nInput:\n'
        prompt += get_grid_str(t['input']) + '\n'
        prompt += 'Output:\n'
        prompt += get_grid_str(t['output']) + '\n'

    prompt += 'Hint: You may want to use the following guidance to implement the function: \n'
    prompt += summarized_hypothesis
    prompt += """
Implement the python function, "transform_grid" enclosed by ```python and ``` according to the hint. Expect to take a numpy array as input. Write as many helper functions as needed.
"""
# Immediately after each helper function, write a test for that helper function. Keep your tests concise and minimal, and ensure they print out helpful information if they fail. Reference the Example 0 by including `from tests import test_example_input, test_example_output` at the top of your code.
    return prompt