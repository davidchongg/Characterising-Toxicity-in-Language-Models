import json
import pandas as pd
from googleapiclient import discovery
from time import sleep
from googleapiclient.errors import HttpError

for sysprompt in range(5):
  df = pd.DataFrame()
  highest_toxicity = 0
  prompt = ""
  response = ""
  for run in range(5):
    read_file = "DataScored/scored_output_sprompt" + str(sysprompt) + "_run" + str(run) + ".jsonl"
    data = pd.read_json(read_file, lines=True)
    df = pd.concat([df, data], axis=0)
  for index, row in df.iterrows():
    if row["toxicity"] > highest_toxicity:
      highest_toxicity = row["toxicity"]
      prompt = row["prompt"]
      response = row["response"]
  
  print(f"Sysprompt: {sysprompt}")
  print(f"Highest toxicity: {highest_toxicity}")
  print(f"Prompt: {prompt}")
  print(f"Response: {response}\n")