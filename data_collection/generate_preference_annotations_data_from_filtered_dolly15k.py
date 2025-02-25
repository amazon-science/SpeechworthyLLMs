from argparse import ArgumentParser
import json 
from speechllm.models import OpenAIGenerativeModel, FalconGenerativeModel
from loguru import logger
from speechllm.utils import form_openai_input_messages, PACKAGE_DIR
import os
from tqdm import tqdm 
import random 
import math
from collections import defaultdict
from speechllm.constants import SYSTEM_PROMPTS

parser = ArgumentParser()
parser.add_argument("--input_file", type=str, default=f"{PACKAGE_DIR}/assets/databricks_dolly_voice_prompt_samples_no_length_limit.jsonl")
parser.add_argument("--output_file", type=str, default=f"{PACKAGE_DIR}/assets/databricks_dolly_voice_prompt_samples_no_length_limit_pref_annotation_responses.jsonl")
parser.add_argument("--llama_model_size", type=str, default="7b", help="one of 70b, 13b, 7b")
parser.add_argument("--falcon_model_size", type=str, default="7b", help="one of 40b, 7b")
parser.add_argument("--partition", "-p", type=int, default=0, help="number from 0 to 9")

args = parser.parse_args()

with open(args.input_file, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

args.output_file = args.output_file.replace(".jsonl", f"_{args.partition}.jsonl")


falcon = FalconGenerativeModel(from_hf_directly=True, model_size=args.falcon_model_size)
gpt4 = OpenAIGenerativeModel(model="gpt-4")
gpt3_5 = OpenAIGenerativeModel(model="gpt-3.5-turbo")

models = [gpt4, gpt3_5, falcon]
temperatures = [0.7, 1.0, 1.3]

sample_candidate_keys = [f"{model.name}_{prompt_key}_response_{temp}" for model in models for prompt_key in SYSTEM_PROMPTS.keys() for temp in temperatures]


random.seed(42)

all_combs = [{"model": model, "temp": temp, "prompt_key": prompt_key} for model in models for prompt_key in SYSTEM_PROMPTS.keys() for temp in temperatures] + [{"model": "response"}]

TOTAL_PREF_SAMPLES_TARGET = 50_000
n_choices = len(all_combs)*(len(all_combs)-1)/2
NUM_PAIRS_PER_SAMPLE = math.floor(TOTAL_PREF_SAMPLES_TARGET/len(data)) + 1 
TARGET_NUM_SAMPLES_EACH_COMB_PAIR = math.floor(len(data) * NUM_PAIRS_PER_SAMPLE / n_choices) + 1 

# choose a priori which pairs to compare for each sample and only generate those. 
comb_pair_counts = defaultdict(int)
all_comb_pairs =[] 
for idx, sample in tqdm(enumerate(data), total=len(data)): 

    sample_comb_pairs = [] 
    while len(sample_comb_pairs) < NUM_PAIRS_PER_SAMPLE:
        comb_pair = random.sample(all_combs, 2)
        comb_pair_name = "_".join(sorted([f"{comb['model'].name}-{comb['prompt_key']}-{comb['temp']}" if comb['model'] != 'response' else 'response' for comb in comb_pair]))
    
        # pass if we have enough samples for this comb pair
        if comb_pair_counts[comb_pair_name] >= TARGET_NUM_SAMPLES_EACH_COMB_PAIR:
            continue

        comb_pair_counts[comb_pair_name] += 1
        sample_comb_pairs.append(comb_pair)
        
    all_comb_pairs.append(sample_comb_pairs)

# divide dataset len by 10
partition_size = math.ceil(len(data)/10)
partition_indices = [i for i in range(partition_size, len(data)+10, partition_size)]
partition_indices[-1] = len(data)
partition_end = partition_indices[args.partition]
partition_start = partition_indices[args.partition-1] if args.partition > 0 else 0


completed_samples = [] 
if os.path.exists(args.output_file):
    with open(args.output_file, "r") as f:
        completed_samples = [json.loads(line) for line in f.readlines()]

start_index = partition_start + len(completed_samples)-1

# generate responses for those for which we will compare. 
for sample, sample_comb_pairs in tqdm(zip(data[start_index:partition_end], all_comb_pairs[start_index:partition_end]), total=len(data[start_index:partition_end])):

    sample_preference_response_pairs = [] 
    for comb_pair in sample_comb_pairs:

        response_key_pairs = [] 
        for config in comb_pair: 

            model = config["model"]
            if model == "response":
                response_key = "response"
                response_key_pairs.append(response_key)
                continue

            temp = config["temp"]
            prompt_key = config["prompt_key"]
            response_key = f"{model.name}_{prompt_key}_response_{temp}"
            response_key_pairs.append(response_key)

            # skip if we already have a response for the given config 
            if response_key in sample: 
                continue 

            conversation_history = form_openai_input_messages(SYSTEM_PROMPTS[prompt_key], sample["instruction"])
            sample[response_key] = model.generate_response(conversation_history, temperature=temp)

        sample_preference_response_pairs.append(response_key_pairs)
    sample["preference_response_pairs"] = sample_preference_response_pairs

    completed_samples.append(sample)
    with open(args.output_file, "w") as f:
        for sample in completed_samples: 
            f.write(json.dumps(sample))
            f.write("\n")

