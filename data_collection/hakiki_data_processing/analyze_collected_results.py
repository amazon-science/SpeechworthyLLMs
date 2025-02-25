import json 
from collections import Counter, defaultdict
from argparse import ArgumentParser
from pprint import pprint 
from loguru import logger
from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np 
from scipy.stats import sem, ttest_ind
from speechllm.utils import HOME_DIR



METRICS_MAPPING = {
    "relevance": {
        "not relevant": 0, 
        "somewhat relevant": 1,
        "relevant": 2,
    },
    "helpful": {
        "not helpful": 0,
        "somewhat helpful": 1,
        "helpful": 2,
        "very helpful": 2,
    },
    "understand": {
        "difficult": 0, 
        "somewhat": 1, 
        "easy": 2,
    },
    "informative": {
        "poor": 0, 
        "fair": 1,
        "good": 2,
        "excellent": 2,
        "too much": 0, 
    },
    "accuracy": {
        "not accurate": 0,
        "contains errors": 1,
        "accurate": 2,
    },
    "length": {
        "too short": 0,
        "short": 1,
        "good": 2,
        "long": 1,
        "too long": 0,
    },
}


def correlation_word_length_and_metrics(data): 

    word_lengths = []
    metrics = {
        "length": [],
        "relevance": [], 
        "understand": [],
        "helpful": [],
        "informative": [],
        "accuracy": [],
    }

    for d in data:
        response = json.loads(d['data']['responses'])[0]
        word_length = len(response['text'].split())

        for asgn in d['assignments']:
            word_lengths.append(word_length)
            for m in metrics.keys():
                metrics[m].append(METRICS_MAPPING[m][asgn['answers'][m].lower()])

    df = pd.DataFrame(metrics)
    df['word_length'] = word_lengths

    print(df.corr("spearman"))
    # print(df.corr())

    return df 


def get_label_counts(data): 
    labels = defaultdict(Counter)
    feedback =[] 
    for d in data: 
        for asgn in d['assignments']: 
            for k, v in asgn['answers'].items(): 
                if 'feedback' in k:
                    if v:  
                        feedback.append(f"{k}: {v}")
                    continue
                labels[k].update([v])

    for k, v in labels.items():
        v = dict(v)

    # print(json.dumps(labels, indent=4))
    return labels 


def plot_barchart(data, title:str =None):

    df = pd.DataFrame(data)

    # Plot
    df.plot(kind='bar', stacked=True, figsize=(12,6))
    if title: 
        plt.title(title)
    else: 
        plt.title('Survey Results')
    plt.ylabel('Number of Responses')
    plt.show() 

def read_jsonl(fn):

    with open(fn, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    return data 

def analyze_single_text_only(fn): 
    data = read_jsonl(fn)

    labels = get_label_counts(data)
    # plot_barchart(labels, "single_text")
    correlation_word_length_and_metrics(data)

    return 

def analyze_single_audio_only(fn): 

    data = read_jsonl(fn)
    labels = get_label_counts(data)
    correlation_word_length_and_metrics(data)

    # plot_barchart(labels, "single_audio")

    return 

def analyze_combined_text_and_audio(fn):

    combined_data = read_jsonl(fn)

    text_results =[] 
    audio_results = [] 
    for d in combined_data: 
        input_data = json.loads(d["data"]["data"])
        hit_id = d["hit"]

        for idx, ind in enumerate(input_data):
            type = ind["type"]
            ind['responses'] = json.dumps(ind['responses'])

            parsed_assignments = [] 
            for asgn in d['assignments']:
                worker_id = asgn['worker_id']
                answers_without_index = {}
                for answer_key in asgn["answers"].keys(): 
                    if f"_{idx}" in answer_key: 
                        key = answer_key.split(f"_{idx}")[0]
                        answers_without_index[key] = asgn["answers"][answer_key]
                parsed_assignments.append({
                    "worker_id": worker_id,
                    "answers": answers_without_index,
                })

            result = {
                "hit": hit_id,
                "data": ind, 
                "assignments": parsed_assignments
            }

            if type =="text": 
                text_results.append(result)
            else:
                audio_results.append(result)


    compare_single_text_and_audio(text_results, audio_results)

def compare_single_text_and_audio(text_data, audio_data):

    text_data = sorted(text_data, key=lambda x: x['data']['instruction'])
    audio_data = sorted(audio_data, key=lambda x: x['data']['instruction'])

    for t_d in text_data: 
        t_d['type'] = 'text'
    for a_d in audio_data: 
        a_d['type'] = 'audio'

    all_data = text_data + audio_data
    text_data_worker_ids = set([a['worker_id'] for d in text_data for a in d['assignments']])
    audio_data_worker_ids = set([a['worker_id'] for d in audio_data for a in d['assignments']])

    intersection = text_data_worker_ids.intersection(audio_data_worker_ids)
    union = text_data_worker_ids.union(audio_data_worker_ids)
    print(f"# workers who did text: {len(text_data_worker_ids)} \n# workers who did audio: {len(audio_data_worker_ids)}")
    print(f"# workers who did both text and audio: {len(intersection)}")
    print(f"# workers who did either text or audio: {len(union)}")
    

    # let's see if normalizing can help 

    # first, get # of assignments per worker
    worker_id_to_num_assignments = defaultdict(Counter)
    total = 0 
    for d in all_data: 
        for asgn in d['assignments']: 
            worker_id_to_num_assignments[asgn['worker_id']][d['type']] += 1
            worker_id_to_num_assignments[asgn['worker_id']]['total'] += 1
            total += 1 

    print(worker_id_to_num_assignments)
    print(total)
    workers_to_consider = [] 
    for worker_id, counts in worker_id_to_num_assignments.items(): 
        if counts['text'] >= 1 and counts['audio'] >= 1:
            workers_to_consider.append(worker_id)
        # if worker_id == "A324VBRLXHG5IB": 
        #     continue 

        # if counts['text'] >= 1 and counts['audio'] < 2 :
        #     workers_to_consider.append(worker_id)

        # if counts['text'] < 2 and counts['audio'] >= 1:
        #     workers_to_consider.append(worker_id)

        # workers_to_consider.append(worker_id)

    print(workers_to_consider)
    # return 

    filter_words = [
        "according to"
    ]

    text_gt_audio = Counter() 
    audio_gt_text = Counter() 
    count = 0 
    threshold = 0.5
    all_text_scores = defaultdict(list)
    all_audio_scores = defaultdict(list)
    word_lengths = [] 
    for t, a in zip(text_data, audio_data):
        # print(t['data']['instruction'])
        # print(a['data']['instruction'])
        instruction = a['data']['instruction']

        if any([filter_word in instruction.lower() for filter_word in filter_words]): 
            continue 

        responses = json.loads(a['data']['responses'])
        # if responses[0]['model'] == 'ground_truth':
        #     continue 

        response_text = responses[0]['text']
        word_length = len(response_text.split())
        # if word_length < 10: 
        #     continue 
        word_lengths.append(word_length)

        count += 1
        assert t['data']['instruction'] == a['data']['instruction']

        text_labels = get_label_counts([t])
        audio_labels = get_label_counts([a])
        text_scores = defaultdict(list)
        audio_scores = defaultdict(list)
        for m in METRICS_MAPPING.keys():
            for asgn in t['assignments']:
                if asgn['worker_id'] not in workers_to_consider:
                    continue 

                if asgn['answers'][m]:
                    text_scores[m].append(METRICS_MAPPING[m][asgn['answers'][m].lower()])
                    all_text_scores[m].append(METRICS_MAPPING[m][asgn['answers'][m].lower()])
            for asgn in a['assignments']:
                if asgn['worker_id'] not in workers_to_consider:
                    continue 

                if asgn['answers'][m]:
                    audio_scores[m].append(METRICS_MAPPING[m][asgn['answers'][m].lower()])
                    all_audio_scores[m].append(METRICS_MAPPING[m][asgn['answers'][m].lower()])

            continue 

            text_scores[m] = sum(text_scores[m]) / len(text_scores[m])
            audio_scores[m] = sum(audio_scores[m]) / len(audio_scores[m])

            # all_text_scores[m].append(text_scores[m])
            # all_audio_scores[m].append(audio_scores[m])

            if text_scores[m] - audio_scores[m] > threshold:
                print(f"text:{text_scores[m]:.2f} vs audio:{audio_scores[m]:.2f} for {m}")
                print(instruction)
                print(response_text)
                # print(text_labels)
                # print(audio_labels)

                text_gt_audio[m] += 1

            if audio_scores[m] - text_scores[m] > threshold:
                print(f"audio:{audio_scores[m]:.2f} vs text:{text_scores[m]:.2f} for {m}")
                print(instruction)
                print(response_text)

                audio_gt_text[m] += 1


    diff_results = defaultdict(dict)
    print(f"# examples with large score diff (text > audio): {text_gt_audio} out of {count}")
    print(f"# examples with large score diff (audio > text): {audio_gt_text} out of {count}")
    for m in METRICS_MAPPING.keys():
        print(f"[{m}] avg. score diff (text - audio): {sum([t - a for t, a in zip(all_text_scores[m], all_audio_scores[m])]) / len(all_text_scores[m]):.2f}")
        average_text_score = np.mean(all_text_scores[m]), sem(all_text_scores[m]) 
        average_audio_score = np.mean(all_audio_scores[m]), sem(all_audio_scores[m])

        print(f"[{m}] avg. text score: {average_text_score[0]:.2f}+-{average_text_score[1]:.2f}")
        print(f"[{m}] avg. audio score: {average_audio_score[0]:.2f}+-{average_audio_score[1]:.2f}")

        # calculate p value for t-test
        t, p = ttest_ind(all_text_scores[m], all_audio_scores[m])
        print(f"[{m}] t-test: t={t:.2f}, p={p:.2f}")

        diff_results[m]['diff (text-audio)'] = round(average_text_score[0] - average_audio_score[0],2)
        diff_results[m]['avg. text score'] = round(average_text_score[0],2)
        diff_results[m]['avg. audio score'] = round(average_audio_score[0],2)
        diff_results[m]['t-test'] = f"t={t:.2f}, p={p:.2f}"

    # breakpoint() 
    text_df = correlation_word_length_and_metrics(text_data)
    audio_df = correlation_word_length_and_metrics(audio_data)

    # plot two line graphs side by side for text & audio that plots word_length vs length 
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # fig.suptitle('Word Length vs. Length')
    # axs[0].plot(text_df['word_length'], text_df['length'], 'o')
    # axs[0].set_title('Text')
    # axs[0].set_xlabel('Word Length')
    # axs[0].set_ylabel('Length')
    # axs[1].plot(audio_df['word_length'], audio_df['length'], 'o')
    # axs[1].set_title('Audio')
    # axs[1].set_xlabel('Word Length')
    # axs[1].set_ylabel('Length')

    # #add regression lines 
    # z = np.polyfit(text_df['word_length'], text_df['length'], 1)
    # p = np.poly1d(z)
    # axs[0].plot(text_df['word_length'],p(text_df['word_length']),"r--")
    # axs[0].annotate(f"y={round(z[0],2)}x+{round(z[1],2)}", xy=(0.05, 0.95), xycoords='axes fraction')

    # z = np.polyfit(audio_df['word_length'], audio_df['length'], 1)
    # p = np.poly1d(z)
    # axs[1].plot(audio_df['word_length'],p(audio_df['word_length']),"r--")
    # # add slope and intercept to plot
    # axs[1].annotate(f"y={round(z[0],2)}x+{round(z[1],2)}", xy=(0.05, 0.95), xycoords='axes fraction')

    # plt.show()

    # plot two bar graphs side by side for text & audio that plots word_length buckets vs counts for each length value 



    diff_results_df = pd.DataFrame(diff_results)
    print(diff_results_df)


    print(word_lengths)
    print(len(word_lengths))
    return

def analyze_single_turn_vs_multiturn(fn):

    data = read_jsonl(fn)

    preference_data = [] 

    text_preference_count = defaultdict(Counter)
    audio_preference_count = defaultdict(Counter)
    for d in data: 
        task_data = json.loads(d["data"]["data"])

        metrics = ["engaging", "understanding", "overall"]
        
        idx2set_type = {idx: each_task_data["set_type"] for idx, each_task_data in enumerate(task_data)}
        idx2conv_type = {idx: each_task_data["conv_type"] for idx, each_task_data in enumerate(task_data)}

        for asgn in d['assignments']:
            worker_id = asgn['worker_id']
            for idx in range(len(task_data)):

                set_type = idx2set_type[idx]
                conv_type = idx2conv_type[idx]
                if conv_type == "single_turn":
                    choice = {"1": "single", "2": "multi"}
                else: 
                    choice = {"1": "multi", "2": "single"}

                for m in metrics:
                    metric_ =  f"{m}_{idx}"
                    if metric_ in asgn['answers']:
                        if set_type == "text":
                            text_preference_count[m][choice[asgn['answers'][metric_]]] += 1 
                        else: 
                            audio_preference_count[m][choice[asgn['answers'][metric_]]] += 1
                        preference_data.append({
                            "worker_id": worker_id,
                            "set_type": set_type,
                            "conv_type": conv_type,
                            "metric": m,
                            "choice": choice[asgn['answers'][metric_]],
                        })
        
            
    print(text_preference_count)
    print(audio_preference_count)

    return 


def main(): 

    parser = ArgumentParser()
    parser.add_argument('--data_file', type=str, help="file that contains the collected datd. e.g. ~/tmp/text_single_prod/collect.json")
    parser.add_argument('--collection_type', '-ct', type=str, help="template used for data collection, one of [single_text, single_audio, single_separate, single_paired, multi_paired]", required=True)
    args = parser.parse_args()

    if args.collection_type == "single_paired": 
        if not args.data_file:
            fn = f"{HOME_DIR}/tmp/combined/collect.json"
        else: 
            fn = args.data_file
        analyze_combined_text_and_audio(fn)

    if args.collection_type == "multi_paired": 
        # fn = f"{HOME_DIR}/tmp/combined_multi2/collect.json"
        if not args.data_file:
            fn = f"{HOME_DIR}/tmp/combined_multi2/collect.json"
        else:
            fn = args.data_file
        analyze_single_turn_vs_multiturn(fn)

    ### Legacy code below ###
    if args.collection_type == "single_text":
        analyze_single_text_only(args.data_file)
    if args.collection_type == "single_audio": 
        analyze_single_audio_only(args.data_file)

    if args.collection_type == "single_separate":

        text_fn = f"{HOME_DIR}/tmp/text_single_prod/collect.json"
        audio_fn = f"{HOME_DIR}/tmp/voice_single_prod/collect.json"

        text_data = read_jsonl(text_fn)
        audio_data = read_jsonl(audio_fn)
        compare_single_text_and_audio(text_data, audio_data)


    return

if __name__ == "__main__": 
    main()