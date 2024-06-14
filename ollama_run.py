import ollama
import pandas as pd
import json
from os import path
import sys
sys.path.insert(1, 'Data')
import toxicity_prompts

data_file_lst = ["Data/nontoxic.jsonl", "Data/toxic.jsonl"]
model_name = "mistral_limit100"

PROMPTS_bak_selected = toxicity_prompts.PROMPTS_bak[::3]

prompt_count = 0
for entry in PROMPTS_bak_selected:
  sys_prompt = entry["sys_prompt"]
  task_desc = entry["task_desc"]
  print("Sys_prompt: " + str(prompt_count))
  for run in range(5):
    print("Run " + str(run))
    output_file = "DataOut/output_sprompt" + str(prompt_count) + "_run" + str(run) + ".jsonl"
    for data_file in data_file_lst:
      with open(data_file, 'r', encoding='utf-8') as file:    ## Iterate for both nontoxic.jsonl and toxic.jsonl
        for i in range(25):     ## Run first 25 samples
          json_object = json.loads(file.readline())
          prompt = task_desc + " " + json_object["prompt"]["text"] 
          response = ollama.generate(model=model_name, prompt=prompt, system=sys_prompt)
          output = response['response']
          data = {"prompt":prompt, "response":output}
          #if i % 1 == 50:
            #print("Done: ", i)
          with open(output_file, 'a', encoding="utf-8")as output:
            json.dump(data, output, ensure_ascii=False)
            output.write("\n")
  prompt_count += 1
