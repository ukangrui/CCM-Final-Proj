import numpy as np
def get_grid_str(grid):
    grid_str = str(np.array(grid))
    return grid_str
import json
with open('logs/id2json.json') as f:
    id2json = json.load(f)
with open('logs/gpt4_0613_100_64.json') as f:
    hypothesis_list = json.load(f)
with open('logs/gpt4_0613_100.json') as f:
    selection = json.load(f)
with open('data/arc-prize-2024/arc-agi_training_challenges.json') as f:
    data = json.load(f)

id2json = {int(k): v.split('.')[0] for k, v in id2json.items()}
example = ""
for i in range(NUM_EXAMPLES):
    try:
        hypothesis = hypothesis_list[i]
        hypothesis_id = hypothesis['task_id']
        hypothesis_name = id2json[hypothesis_id]
        selection_id = selection[str(hypothesis_id)][0]
        true_hypothesis = hypothesis['results'][selection_id]['parsed']
        example += f'Example {i+1}\n'
        task = data[hypothesis_name]
        for j, t in enumerate(task['train']):
            example += f'Case {j}:\nInput:\n'
            example += get_grid_str(t['input']) + '\n'
            example += 'Output:\n'
            example += get_grid_str(t['output']) + '\n'
        example += f'Describing the input grid: {true_hypothesis['description_input']}\n'
        example += f'Describing the size of the output grid: {true_hypothesis['description_output_grid_size']}\n'
        example += f'Describing how to transform the grid: {true_hypothesis['description_output']}\n'
    except:
        pass