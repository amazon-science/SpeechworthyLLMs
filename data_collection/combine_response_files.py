import json 
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm 
from speechllm.utils import add_audio_for_sample, PACKAGE_DIR
import os 
import functools

import concurrent.futures

def process_sample(sample, sample_key_to_overwrite=None):
    return add_audio_for_sample(sample, overwrite=False, sample_key_to_overwrite=sample_key_to_overwrite)


parser = ArgumentParser()
parser.add_argument("--input_file_base", type=str, default=f"{PACKAGE_DIR}/assets/databricks_dolly_voice_prompt_samples_no_length_limit_pref_annotation_responses_")
parser.add_argument("--output_file", type=str, default=f"{PACKAGE_DIR}/assets/databricks_dolly_voice_prompt_samples_no_length_limit_pref_annotation_responses.jsonl")
parser.add_argument("--test", action="store_true")
parser.add_argument("--parallel", action="store_true")
parser.add_argument("--sample_key_to_overwrite", "-sk", type=str, default=None)

args = parser.parse_args()

data = []

for i in range(10):
    fp = Path(args.input_file_base + str(i) + ".jsonl")
    with open(fp, "r") as f:
        data.extend([json.loads(line) for line in f.readlines()])

for sample in data: 
    for key, value in sample.items():
        if "response" in key and "falcon" in key: 
            if not isinstance(value, str): 
                response = value[0]["generated_text"]
                sample[key] = response.split("\nUser")[0]


data_with_audio = [] 


if args.test:
    args.output_file = args.output_file.replace(".jsonl", "_test.jsonl")
    data = data[:10]

if args.parallel: 
    # Define the number of processes you want to spawn
    num_processes = 4

    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        partial_process = functools.partial(process_sample, sample_key_to_overwrite=args.sample_key_to_overwrite)
        results = list(tqdm(executor.map(partial_process, data), total=len(data)))
    data_with_audio.extend(results)

else: 
    for sample in tqdm(data, total=len(data)):
        sample = add_audio_for_sample(sample, overwrite=False, sample_key_to_overwrite=args.sample_key_to_overwrite)
        data_with_audio.append(sample)


with open(args.output_file, "w") as f:
    for sample in data_with_audio:
        f.write(json.dumps(sample))
        f.write("\n")

