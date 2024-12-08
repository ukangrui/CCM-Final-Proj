import os
import subprocess
import tempfile
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import enum
import numpy as np
import tempfile
import os
import subprocess
import textwrap
import sys
import re
from ast import literal_eval
### Judge takes input - output pairs for each problem

### Judge Utils
def percentage_correct(grid, predicted_grid):
    pass


class ARCJudge:
    def __init__(self, tmp_dir = './tmp'):
        self.tmp_dir = tmp_dir

    def execute_function(self, func_code: str, input_array):
        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)
        try:
            temp_code_path = os.path.join(self.tmp_dir, 'temp_function.py')
            temp_array_path = os.path.join(self.tmp_dir, 'input_array.npy')
            wrapper_code = textwrap.dedent(f"""
import numpy as np
{func_code}
input_array = np.load('{temp_array_path}')
result = transform_grid(input_array)
print(result)
            """)
            with open(temp_code_path, 'w+') as temp_code_file:
                temp_code_file.write(wrapper_code)
            np.save(temp_array_path, input_array)
            process = subprocess.Popen(
                ['python3', temp_code_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                return "Run Error"
            try:
                stdout = self.format_matrix_string(stdout)
                return stdout
            except:
                return "Parse Error"
        finally:
            if os.path.exists(temp_code_path):
                os.remove(temp_code_path)
            if os.path.exists(temp_array_path):
                os.remove(temp_array_path)
        
    def percentage_correct(self, output: np.ndarray, expected_output:np.ndarray): 
        if output.shape != expected_output.shape:
            return 0.0
        else:
            mask = (output == expected_output)
            return mask.sum() / mask.size

    def format_matrix_string(self, matrix_string):
        formatted = matrix_string.replace('\n', '').replace(' ', '')
        formatted = formatted.strip('[]')
        rows = formatted.split('][')
        formatted_rows = []
        for row in rows:
            formatted_row = ','.join(row.split(']')[0].split('[')[-1])
            formatted_rows.append(f'[{formatted_row}]')
        return np.array(literal_eval(f'[{",".join(formatted_rows)}]'))