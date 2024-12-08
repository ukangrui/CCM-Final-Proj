### Pipeline: 
1) generates a set of candidate hypotheses     output: {training example, list [hypothesis]}
2) selects a subset (summarize [hypothesis])   output: {training example, set  [hypothesis]}
3) implements each hypothesis in code as a function output: {training example, set  [hypothesis], set[list [programs]]}
4) validates the implementations against the training examples. -> metric: percentage correct
5) [optional] self [reflection] on generated programs, fix for 3 steps, skip for now
6) return training_example - best hypothesis pair


### Args:
1) prompt: generate_prompt, summarize_prompt, impelment_prompt(minimal change)
2) original scale:  num_hypothesis: 64, num_summarized_hypothesis: 8, num programs: 64
3) test small scale (GPT): num_hypothesis: 5, num_summarized_hypothesis: 2, num programs: 5, 10 test problems


### Notes:
1) The original paper do not open source the code execution part, had to build our own