from speechllm.speechllm_benchmark import SpeechLLMEval
from speechllm.utils import PACKAGE_DIR
import json 
from loguru import logger 
import pandas as pd
import os 

vllm_benchmark = SpeechLLMEval()

responses_fn = f"{PACKAGE_DIR}/human_evaluation_responses_intermediate.jsonl"

with open(responses_fn, "r") as f:
    responses = [json.loads(line) for line in f.readlines()]


# group outputs by models 
model2responses = {}
for r in responses: 
    model = r["model"]
    if model not in model2responses:
        model2responses[model] = []
    model2responses[model].append(r)


# for each model, compute the average score
model_scores = [] 
for model, responses in model2responses.items():

    scores = vllm_benchmark.score_responses(responses)
    for score in scores: 
        model_scores.append({
            "model": model,
            **score
        })


model_scores = pd.DataFrame(model_scores)

statistics = model_scores.groupby("model").agg(["mean", "sem"])


# remove truncation in pandas
pd.set_option('display.max_colwidth', None)
print(statistics)

# save to csv 
statistics.to_csv(f"{PACKAGE_DIR}/automatic_evaluation_scores.csv", index=False)

order = ["length", "voice_suitability", "information_completeness", "relevance","understandability", "dependency_graph_depth"]
for idx, row in statistics.iterrows():
    overleaf_string = ' & '.join([f"{row[metric]['mean']:.1f}" + "_{" + f"{row[metric]['sem']:.1f}" + "}" for metric in order])
         
    overleaf_string = overleaf_string.replace("1.0_{0.0}", "-")
    print(row.name, overleaf_string)
