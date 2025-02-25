from argparse import ArgumentParser
import json
import base64
import requests
import random
from speechllm.utils import cleanse_for_json, transform_audio_to_base64, PACKAGE_DIR

parser = ArgumentParser()
parser.add_argument('--input_file', type=str, default=f'{PACKAGE_DIR}/assets/databricks_dolly_voice_prompt_samples.jsonl')
parser.add_argument('--base64', action='store_true')
args = parser.parse_args()


with open(args.input_file, 'r') as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

new_multi_response_format_data = [] 
new_single_response_format_data = [] 

for d in data:

    models = [] 
    for key in d.keys(): 
        if 'decomposed' in key: 
            continue 
        if len(key.split('-response')) > 1: 
            models.append(key.split('-response')[0])

    models = list(set(models))

    has_as_an_ai_response = False 

    responses = [
        {
            "model": "ground_truth", 
            "text": cleanse_for_json(d['response']),
            "audio_url": transform_audio_to_base64(d['response_audio_url']) if args.base64 else d['response_audio_url']
        }
    ]
    for model in models: 
        responses.append(
            {
                "model": model,
                "text": cleanse_for_json(d[model + '-response']),
                "audio_url": transform_audio_to_base64(d[model + '-response_audio_url']) if args.base64 else d[model + '-response_audio_url']
            }            
        )

        if "ai language model" in d[model + '-response'].lower(): 
            has_as_an_ai_response = True
    
    if has_as_an_ai_response: 
        continue 

    # assign non-overlapping random integers to each response within the range of len(responses)
    random_ints = list(range(len(responses)))
    random.shuffle(random_ints)
    for idx, response in enumerate(responses):
        response['order'] = random_ints[idx]

    response_json_text = json.dumps(responses)


    new_multi_response_sample = {
        "category": d['category'],
        "context": d['context'],
        "instruction": d['instruction'],
        "instruction_audio_url": transform_audio_to_base64(d['instruction_audio_url']) if args.base64 else d['instruction_audio_url'],
        "responses": response_json_text
    }

    for response in responses: 
        if response["model"] == "ground_truth": 
            continue
        
        new_single_response_sample = {
            "category": d['category'],
            "context": d['context'],
            "instruction": d['instruction'],
            "instruction_audio_url": transform_audio_to_base64(d['instruction_audio_url']) if args.base64 else d['instruction_audio_url'],
            "responses": json.dumps([response])
        }
        new_single_response_format_data.append(new_single_response_sample)

    new_multi_response_format_data.append(new_multi_response_sample)

with open(f"{PACKAGE_DIR}/assets/mturk_inputs_multi_response.jsonl", 'w') as f:
    for idx, d in enumerate(new_multi_response_format_data):
        # if idx == 40: 
        #     break 
        f.write(json.dumps(d) + "\n")

with open("mturk_inputs_single_response.jsonl", 'w') as f:
    random.shuffle(new_single_response_format_data)
    for idx, d in enumerate(new_single_response_format_data):
        # if idx == 40: 
        #     break 
        f.write(json.dumps(d) + "\n")


# small sample
with open(f"{PACKAGE_DIR}/assets/mturk_inputs_multi_response_small.jsonl", 'w') as f:
    for idx, d in enumerate(new_multi_response_format_data):
        if idx == 10: 
            break 
        f.write(json.dumps(d) + "\n")

with open(f"{PACKAGE_DIR}/assets/mturk_inputs_single_response_small.jsonl", 'w') as f:
    for idx, d in enumerate(new_single_response_format_data):
        if idx == 10: 
            break 
        f.write(json.dumps(d) + "\n")



