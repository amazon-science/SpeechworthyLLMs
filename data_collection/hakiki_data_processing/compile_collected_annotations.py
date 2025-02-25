import json
from loguru import logger
from collections import defaultdict
from speechllm.utils import PACKAGE_DIR, HOME_DIR

fns= [
    f"{HOME_DIR}/tmp/audio_pref_production_last_5000/collect.json",
    f"{HOME_DIR}/tmp/audio_pref_production_1000/collect.json",
]

newer_fns = [
    f"{HOME_DIR}/tmp/audio_pref_production_last10k-5k/collect.json",
    f"{HOME_DIR}/tmp/audio_pref_production_batch0/collect.json",
    f"{HOME_DIR}/tmp/audio_pref_production_batch1/collect.json",
    f"{HOME_DIR}/tmp/audio_pref_production_batch2/collect.json",
]

fns = fns + newer_fns

collected_data = []
workers_count = defaultdict(int)
workers_word_count = defaultdict(int)
worker2assignment = {}
annotations_with_bugs = [] 
unique_instructions = [] 

blacklisted_workers = [
    ""
]

for fn in fns:
    with open(fn, "r") as f:
        for line in f.readlines():
            annotation_sample = json.loads(line)
            annotation_input_data = json.loads(annotation_sample["data"]["data"])

            if not annotation_sample["assignments"]:
                continue 

            # breakpoint() 

            for assgn in annotation_sample["assignments"]:

                worker_id = assgn["worker_id"]
                workers_count[worker_id] += 1
                assignment_id = assgn["assignment_id"]
                worker2assignment[worker_id] = assignment_id
                if worker_id in blacklisted_workers:
                    continue

                if 'answers' not in assgn: 
                    annotations_with_bugs.append(annotation_sample)
                    continue 

                if 'overall_0' not in assgn['answers']:
                    annotations_with_bugs.append(annotation_sample)
                    continue

                explanation = assgn["answers"]["question-followup"]
                workers_word_count[worker_id] += len(explanation.split())


                if assgn['answers']['overall_0'] == "Response 1":
                    chosen = "model_0"
                    rejected = "model_1"
                else:
                    chosen = "model_1"
                    rejected = "model_0"

                formatted_data = {
                    "instruction":  annotation_input_data['instruction'],
                    "chosen": annotation_input_data[f"{chosen}_response"],
                    "rejected": annotation_input_data[f"{rejected}_response"],
                    "chosen_model": annotation_input_data[chosen],
                    "rejected_model": annotation_input_data[rejected],
                    "explanation": assgn['answers']['question-followup'],
                    "margin": assgn['answers']['information_0'],
                }                

                if formatted_data['instruction'] not in unique_instructions:
                    unique_instructions.append(formatted_data['instruction'])

                collected_data.append(formatted_data)


for worker_id, count in workers_count.items():
    if count < 100: 
        continue 
    if worker_id in blacklisted_workers:
        continue
    bonus = count*0.12
    print(f"{worker_id} | {count} samples | avg. {workers_word_count[worker_id]/count:.2f} words | bonus: {bonus:.2f} | assignment_id: {worker2assignment[worker_id]}")
    print(f"aws mturk send-bonus --worker-id {worker_id} --assignment-id {worker2assignment[worker_id]} --bonus-amount {bonus:.2f} --reason 'Thank you for your work on the audio preference task. Bonus $.12 for every completed HIT.' --profile mturk")

logger.info(f"Total number of samples annotated: {len(collected_data)}")
logger.info(f"Total number of workers: {len(workers_count)}")
logger.info(f"Total number of annotations with bugs: {len(annotations_with_bugs)}")
logger.info(f"Total number of unique instructions: {len(unique_instructions)}")

# split by instructins into train set and test set 
train_set_instructions = unique_instructions[:int(len(unique_instructions)*0.9)]
test_set_instructions = unique_instructions[int(len(unique_instructions)*0.9):]

train_set = []
test_set = []

for sample in collected_data:
    if sample['instruction'] in train_set_instructions:
        train_set.append(sample)
    else:
        test_set.append(sample)

logger.info(f"Total number of samples in train set: {len(train_set)}")
logger.info(f"Total number of samples in test set: {len(test_set)}")

output_fn = f"{PACKAGE_DIR}/assets/speech_preference_data/train.jsonl"
with open(output_fn, "w") as f:
    for sample in train_set:
        f.write(json.dumps(sample) + "\n")

output_fn = f"{PACKAGE_DIR}/assets/speech_preference_data/test.jsonl"
with open(output_fn, "w") as f:
    for sample in test_set:
        f.write(json.dumps(sample) + "\n")

        