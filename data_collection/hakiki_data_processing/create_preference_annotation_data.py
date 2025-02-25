from argparse import ArgumentParser
import json
import os 
import random 
from speechllm.utils import cleanse_for_json, PACKAGE_DIR, HOME_DIR
from loguru import logger


parser = ArgumentParser()
parser.add_argument("--input_file", type=str, default=f"{PACKAGE_DIR}/assets/databricks_dolly_voice_prompt_samples_no_length_limit_pref_annotation_responses_no_duplicates.jsonl")
parser.add_argument("--output_file", type=str, default=f"{PACKAGE_DIR}/assets/mturk_inputs_single_turn_preference_annotation.jsonl")
parser.add_argument("--subset", action="store_true")
parser.add_argument("--start_index", "-s", type=int, default=0)
parser.add_argument("--end_index", "-e", type=int, default=100)
parser.add_argument("--use_interval", "-ui", action="store_true")
parser.add_argument("--interval", "-i", type=int, default=5000)

args = parser.parse_args()

with open(args.input_file, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

already_collected = [] 
files = [
    f"{HOME_DIR}audio_pref_production_last10k-5k/collect.json",
    f"{HOME_DIR}audio_pref_production_last_5000/collect.json", 
    f"{HOME_DIR}audio_pref_production_1000/collect.json",
    f"{PACKAGE_DIR}/assets/mturk_inputs_single_turn_preference_annotation_last5000-0.jsonl",
    f"{PACKAGE_DIR}/assets/mturk_inputs_single_turn_preference_annotation_last10000-5001.jsonl"
]
for fn in files: 
    with open(fn, "r") as f:
        
        for line in f.readlines():
            annotation_sample = json.loads(line)
            if "collect.json" in fn: 
                if "assignments" in annotation_sample and annotation_sample["assignments"]: 
                    already_collected.append(annotation_sample["data"]["data"])

            else: 
                already_collected.append(annotation_sample["data"])

logger.info(f"Total number of samples already collected: {len(set(already_collected))}")
breakpoint() 

mturk_data = []
for sample in data:
    preference_pairs = sample['preference_response_pairs'] 
    for preference_pair in preference_pairs:
        mturk_sample = {
            "instruction": cleanse_for_json(sample["instruction"]), 
            "instruction_audio_url": sample["instruction_audio_url"],
        }

        random.shuffle(preference_pair)

        for idx, response_type in enumerate(preference_pair):

            mturk_sample[f"model_{idx}"] = response_type
            mturk_sample[f"model_{idx}_response"] = cleanse_for_json(sample[response_type]) 
            mturk_sample[f"model_{idx}_audio_url"] = sample[response_type + "_audio_url"]

        # if any value is None, skip. these are cases where the audio file was not created due to exceeding the permissible maximum length. 
        if any([v is None for v in mturk_sample.values()]):
            continue

        mturk_sample['set_type'] = "audio"
        annotation_input_data = json.dumps(mturk_sample)
        # breakpoint() 
        if annotation_input_data in already_collected:
            continue
        mturk_data.append({"data": annotation_input_data})

        # if args.subset: 
        #     break
    # breakpoint()

logger.info("Total number of samples remaining to collect: {}".format(len(mturk_data)))

if args.subset: 
    if args.end_index == 0:
        args.end_index = len(mturk_data)
    subset_data = mturk_data[args.start_index:args.end_index]
    logger.info("Total number of samples: {}".format(len(subset_data)))
    output_file = args.output_file.replace(".jsonl", f"_{args.start_index}_{args.end_index}.jsonl")

    with open(output_file, "w") as f:
        for sample in subset_data:
            f.write(json.dumps(sample) + "\n")

if args.use_interval: 
    batch_index = 0 
    for i in range(0, len(mturk_data), args.interval):
        output_file = args.output_file.replace(".jsonl", f"_batch{batch_index}.jsonl")
        interval_data = mturk_data[i: min(i+args.interval, len(mturk_data))]
        logger.info("Total number of samples: {}".format(len(interval_data)))
        with open(output_file, "w") as f:
            for sample in interval_data:
                f.write(json.dumps(sample) + "\n")

        batch_index += 1

