# combine single responses to create pairs that have similar word counts in the text response

import json 
import random
from speechllm.utils import PACKAGE_DIR

fn = f"{PACKAGE_DIR}/assets/mturk_inputs_single_response.jsonl"
MAX_N_SAMPLES = 80

with open(fn, 'r') as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

def get_word_length(text): 

    return len(text.split())

for d in data: 
    d['word_length'] = get_word_length(json.loads(d['responses'])[0]['text'])

data = sorted(data, key=lambda x: x['word_length'])
data = [d for d in data if d['word_length'] > 10]

for d in data: 
    d['responses'] = json.loads(d['responses'])

paired_data = [] 
for idx in range(0, len(data), 2): 

    # if first and second have same instructions, skip 
    if data[idx]['instruction'] == data[idx + 1]['instruction']:
        continue 

    data[idx]['type'] = 'audio'
    data[idx + 1]['type'] = 'text'

    paired_data.append(
        {
            "data": json.dumps([data[idx], data[idx + 1]])
        }
    )

    data[idx]['type'] = 'text'
    data[idx + 1]['type'] = 'audio'

    paired_data.append(
        {
            "data": json.dumps([data[idx], data[idx + 1]])
        }
    )

paired_data = paired_data[:MAX_N_SAMPLES]
random.shuffle(paired_data)

print(f"# of pairs: {len(paired_data)}")

with open(f"{PACKAGE_DIR}/assets/mturk_inputs_single_response_paired.jsonl", 'w') as f:
    for d in paired_data:
        f.write(json.dumps(d) + "\n")

with open(f"{PACKAGE_DIR}/assets/mturk_inputs_single_response_paired_test.jsonl", 'w') as f:
    for idx, d in enumerate(paired_data):
        if idx == 10: 
            break 
        f.write(json.dumps(d) + "\n")

