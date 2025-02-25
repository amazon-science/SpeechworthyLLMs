from argparse import ArgumentParser
from loguru import logger
import random 
import json 
from speechllm.utils import cleanse_for_json, PACKAGE_DIR

parser = ArgumentParser()
parser.add_argument('--input_file', type=str, default=f'{PACKAGE_DIR}/assets/databricks_dolly_voice_prompt_samples.jsonl')
args = parser.parse_args()

with open(args.input_file, 'r') as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

new_multi_turn_data = [] 

data_with_multiturns =[] 
for d in data: 

    d['gpt-4-response'] = cleanse_for_json(d['gpt-4-response'])
    d['instruction'] = cleanse_for_json(d['instruction'])
    d['response'] = cleanse_for_json(d['response'])
    d['gpt-3.5-turbo-response'] = cleanse_for_json(d['gpt-3.5-turbo-response'])

    if "as an ai" in d['gpt-4-response'].lower():
        continue

    decomposed_responses = d.get("decomposed-gpt-4-response")
    if decomposed_responses and not isinstance(decomposed_responses, str): 

        decomposed_responses = [cleanse_for_json(r['text']) for r in decomposed_responses]

        if decomposed_responses[0][:-1].lower() != d['instruction'][:-1].lower():
            logger.info(f"Instruction does not match the first turn of the conversation:\n\t{d['instruction']}\n\t{decomposed_responses[0]}")
            continue

        if len(decomposed_responses) > 12:
            continue 

        data_with_multiturns.append({
                "data": d
            }
        )

# order by number of turns in the conversation 
data_with_multiturns = sorted(data_with_multiturns, key=lambda x: len(x['data']['decomposed-gpt-4-response']))

# group by twos 
for idx in range(0, len(data_with_multiturns), 2):
    if idx + 1 > len(data_with_multiturns) - 1:
        break

    # randomly choose between text and audio 
    set_type1 = random.choice(["text", "audio"])
    set_type2 = "audio" if set_type1 == "text" else "text"

    # randomly choose between single turn and muli turn 
    conv_type1 = random.choice(["single_turn", "multi_turn"])
    conv_type2 = "multi_turn" if conv_type1 == "single_turn" else "single_turn"

    data_with_multiturns[idx]['data']['set_type'] = set_type1
    data_with_multiturns[idx+1]['data']['set_type'] = set_type2
    data_with_multiturns[idx]['data']['conv_type'] = conv_type1
    data_with_multiturns[idx+1]['data']['conv_type'] = conv_type2

    new_multi_turn_data.append({
        "data": json.dumps([
            data_with_multiturns[idx]['data'],
            data_with_multiturns[idx+1]['data']
        ])
    })


logger.info(f"# of pairs of samples with multiturns: {len(new_multi_turn_data)}")

with open(f"{PACKAGE_DIR}/assets/mturk_inputs_multiturn.jsonl", 'w') as f:
    for d in new_multi_turn_data:
        f.write(json.dumps(d) + "\n")

with open(f"{PACKAGE_DIR}/assets/mturk_inputs_multiturn_test.jsonl", 'w') as f:
    for idx, d in enumerate(new_multi_turn_data):
        if idx == 10:
            break 
        f.write(json.dumps(d) + "\n")
                




    
