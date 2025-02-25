import json 
from argparse import ArgumentParser
import pandas as pd 
from collections import OrderedDict, defaultdict
from pprint import pprint

parser = ArgumentParser()
parser.add_argument('--input_file', type=str, default="~/project/tmp/olmo_comp/collect.json")
parser.add_argument("--drop_temperature", "-d", help="Drop temperature in key names", action="store_true")
parser.add_argument("--count_ties", "-c", help="count ties", action="store_true")
parser.add_argument("--only_compare_truths", "-o", help="only compare truths", action="store_true")
parser.add_argument("--compare_truths_only_or_both_false", "-b", help="only compare truths or both false", action="store_true")
args = parser.parse_args()

# read jsonl file with pandas
df = pd.read_json(args.input_file, lines=True)

comparison_results = {}

def shorten_name(name): 

    name = name.replace("_response", "")
    name = name.replace("response_", "")
    name = name.replace("gpt-3.5-turbo", "gpt-3.5")
    name = name.replace("falcon-7b-instruct", "falcon")

    return name 

all_model_types = set()
factuality_results = {}

labels = [] 
for idx, row in df.iterrows(): 

    input_data = json.loads(row['data']['data'])

    if row['assignments']: 
        assignment_labels = [] 
        for assignment in row['assignments']:
            if 'overall_0' not in assignment['answers']:
                continue 

            answer = assignment['answers']['overall_0']
            if answer == "Response 1": 
                chosen = "model_0"
                rejected = "model_1"
            else: 
                chosen = "model_1"
                rejected = "model_0"

            chosen_model = input_data[chosen]
            rejected_model = input_data[rejected]
            all_model_types.add(chosen_model)
            all_model_types.add(rejected_model)
            
            if 'question-followup' in assignment['answers']:
                explanation = assignment['answers']['question-followup']
            elif 'explanation' in assignment['answers']:
                explanation = assignment['answers']['explanation']

            response_1_factual = assignment['answers'].get('response_1_factual_0')
            response_2_factual = assignment['answers'].get('response_2_factual_0')

            if response_1_factual is not None and response_2_factual is not None:
                if chosen_model not in factuality_results:
                    factuality_results[chosen_model] = defaultdict(int)
                if rejected_model not in factuality_results:
                    factuality_results[rejected_model] = defaultdict(int)

                if chosen == 'model_0':
                    factuality_results[chosen_model][response_1_factual]+=1
                    factuality_results[rejected_model][response_2_factual]+=1
                else: 
                    factuality_results[chosen_model][response_2_factual]+=1
                    factuality_results[rejected_model][response_1_factual]+=1 

                if args.only_compare_truths: 
                    if response_1_factual != "Yes" or response_2_factual != "Yes": 
                        continue 

                if args.compare_truths_only_or_both_false: 
                    if (response_1_factual != "Yes" or response_2_factual != "Yes") and not (response_1_factual == "No" and response_2_factual == "No"): 
                        continue


            if args.drop_temperature:
                chosen_model = '_'.join(chosen_model.split("_")[:-1])
                rejected_model = '_'.join(rejected_model.split("_")[:-1])
                if rejected_model == "": 
                    rejected_model = "response"
                if chosen_model == "":
                    chosen_model = "response"

            chosen_model = shorten_name(chosen_model)
            rejected_model = shorten_name(rejected_model)

            combined_name = '-vs-'.join(sorted([chosen_model, rejected_model]))
            if combined_name not in comparison_results:
                comparison_results[combined_name] = defaultdict(int)

            margin = assignment['answers'].get('margin_0', None)
            if margin == 'Negligibly better'  and args.count_ties: 
                chosen_model = 'tie'


            comparison_results[combined_name][chosen_model] += 1
            assignment_labels.append(chosen_model) 
        labels.append(assignment_labels)
# show agreement between annotators
breakpoint() 
agreements = 0 
num_samples_with_multiple = 0 
disagreement_counts = defaultdict(int)
for label in labels:
    if len(label) > 1: 
        num_samples_with_multiple += 1
        # if label[0] == label[1] or label[0] == "tie" or label[1] == "tie": 
        if label[0] == label[1]: 
            agreements += 1
        else: 
            disagreement_counts[tuple(sorted(label))] += 1
agreement_rate = agreements / num_samples_with_multiple
print(f"Agreement rate: {agreement_rate:.2f}")
print("Disagreement counts:")
pprint(dict(disagreement_counts))

# print(comparison_results)
results_to_show = [] 
for comparison_key, results in comparison_results.items(): 
    # print those that have 0 as value
    delta = 0 
    keys = results.keys()
    vals = list(results.values())

    # if len(keys) < 2:
    #     continue

    # if abs(vals[0] - vals[1]) >= delta:
    # # if any([v ==0 for v in list(results.values())]): 
    #     # 
    #     # print(comparison_key)
    #     # print(results)
    #     # print()

    results_to_show.append({
        comparison_key: results
    })



# order results_to_show by key 
results_to_show = sorted(results_to_show, key=lambda x: list(x.keys())[0])
# print(results_to_show)

for result in results_to_show: 
    result_key = list(result.keys())[0]
    result_val = result[result_key]
    # pprint(dict(result_val))

    # change to % 
    total = sum(result_val.values())
    print(f"{result_key} ({total} annotations)")
    for k, v in result_val.items():
        result_val[k] = f"{v/total*100:.1f}%"
    pprint(dict(result_val))

# print(comparison_results)
# pprint(factuality_results)

# get percentage of false statements
for model, results in factuality_results.items():
    total = sum(results.values())
    print(f"{model} ({total} annotations)", end=": ")
    false_pct = (results['No'] + results['Partially factual']) / total * 100
    print(f"False statements: {false_pct:.1f}%")