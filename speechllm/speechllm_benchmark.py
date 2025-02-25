import textstat
import spacy 
import numpy as np 
from loguru import logger 
from typing import List, Tuple
from argparse import ArgumentParser
import pandas as pd 
import json
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from pathlib import Path
import hashlib
from pprint import pprint 

# custom imports
from speechllm.models import create_reward_fn
from speechllm.utils import format_input_string_for_reward_model, format_prompt_response_for_reward_model, add_audio_for_text, cleanse_for_json, PACKAGE_DIR
from speechllm.dataset_utils import is_voice_suitable_prompt_based_on_keywords

HOME_DIR = os.environ["HOME"]

class SpeechLLMEval: 

    def __init__(self, args=None, speech_reward_model_path:str = f"{PACKAGE_DIR}/assets/gptj_reward_model") -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.args = args

        # self.voice_suitability_model = 
        if speech_reward_model_path: 
            if not os.path.exists(speech_reward_model_path):
                raise ValueError(f"Model path does not exist: {speech_reward_model_path}")
            self.reward_fn = create_reward_fn(speech_reward_model_path)
        else: 
            self.reward_fn = None

    def get_readability_score(self, text:str) -> float: 
        return textstat.flesch_reading_ease(text)

    def get_dependency_graph_depth(self, text:str) -> Tuple[float, List[int]]:

        def depth_of_tree(token):
            # Depth of a tree is max depth of children + 1
            if not list(token.children):
                return 1
            else:
                return max(depth_of_tree(child) for child in token.children) + 1

        # Apply dependency parsing
        doc = self.nlp(text)

        depths = [] 
        for sent in doc.sents:
            root = sent.root
            depth = depth_of_tree(root)            

            # logger.info(f"text: {text}")
            # logger.info(f"depth: {depth}")            

            depths.append(depth)
        
        mean_depth = np.mean(depths)
        
        return mean_depth, depths
    

    def get_word_count(self, text): 
        
        return len(text.split())
    
    def get_proportion_of_unvocalizable_content(self, text:str): 

        nonvocalizable_content = "/\{([-])}|*_^&%$#@~`"
        list_items = ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.", "11.", "12.", "13.", "14.", "15.", "16.", "17.", "18.", "19.", "20."]
        count = 0
        for char in text:
            if char in nonvocalizable_content:
                count += 1
        for item in list_items:
            if item in text:
                count += 1
        
        return count 
    
    def get_top_tf_idf_terms(self, text_batch:List[str]): 

        tf_idf = TfidfVectorizer()
        tf_idf.fit_transform(text_batch)

        feature_names = tf_idf.get_feature_names_out()
        return feature_names

    def get_voice_suitability_score_single(self, user_instruction:str, response:str):

        samples = [format_prompt_response_for_reward_model(user_instruction, response)]
        if self.reward_fn:
            rewards = self.reward_fn(samples).tolist()
            return rewards[0]
        return -10000


    def batch_evaluate(self, user_instructions, responses):

        samples = [format_prompt_response_for_reward_model(user_instruction, response) for user_instruction, response in zip(user_instructions, responses)]

        logger.info(f"Starting batch evaluation for {len(samples)} samples")

        batch_scores = {
            "read.": [self.get_readability_score(response) for response in responses],
            "depend.": [self.get_dependency_graph_depth(response)[0] for response in responses],
            "voice_rm": self.reward_fn(samples).tolist() if self.reward_fn else [-10000] * len(samples),
            "wc": [self.get_word_count(response) for response in responses],
            "nonvocal": [self.get_proportion_of_unvocalizable_content(response) for response in responses],
        }

        logger.info(f"Batch evaluation completed")
        return batch_scores

    def evaluate_turn(self, user_instruction:str, response:str):

        scores = {
            "read.": self.get_readability_score(response),
            "depend.": self.get_dependency_graph_depth(response)[0],
            "voice_rm": self.get_voice_suitability_score_single(user_instruction, response),
            "wc": self.get_word_count(response),
            "nonvocal": self.get_proportion_of_unvocalizable_content(response),
        }
        return scores

    
    def print_stats(self, scores): 

        max = np.max(scores)
        min = np.min(scores)
        mean = np.mean(scores)
        std = np.std(scores)
        median = np.median(scores)

        return f"max: {max}, min: {min}, mean: {mean}, std: {std}, median: {median}"

    def print_single_vs_multi(self, instances): 
        single_response_readability_scores = []
        single_response_dependency_graph_depths = []

        decomposed_readability_scores = []
        decomposed_dependency_graph_depths = []

        for instance in tqdm(instances): 

            single_response_readability_scores.append(self.get_readability_score(instance['response']))
            single_response_dependency_graph_depths += self.get_dependency_graph_depth(instance['response'])[1]

            decomposed = instance.get("decomposed-gpt-4-response", [])
            for turn in decomposed[1::2]: 
                decomposed_readability_scores.append(self.get_readability_score(turn['text']))
                decomposed_dependency_graph_depths += self.get_dependency_graph_depth(turn['text'])[1]


        print(f"Single response readability score: {self.print_stats(single_response_readability_scores)}")
        print(f"Single response dependency graph depth: {self.print_stats(single_response_dependency_graph_depths)}")

        print(f"Decomposed readability score: {self.print_stats(decomposed_readability_scores)}")
        print(f"Decomposed dependency graph depth: {self.print_stats(decomposed_dependency_graph_depths)}")

        return    
    
    def print_reference_examples(self, instances, n_examples=1):

        for instance in instances[:n_examples]:
            print(f"User instruction: {instance['instruction']}")
            print(f"Response: {instance['response']}")


            print(f"{'='*50}")

            decomposed = instance.get("decomposed-gpt-4-response", [])
            decomposed_readability_scores = [] 
            decomposed_dependency_graph_depths = []
            for turn in decomposed[1::2]: 
                print(f"Decomposed turn: {turn['text']}")
                readability_score = self.get_readability_score(turn['text'])
                decomposed_readability_scores.append(readability_score)

                dependency_graph_mean_depth, dependency_graph_depths = self.get_dependency_graph_depth(turn['text'])
                decomposed_dependency_graph_depths += dependency_graph_depths

            print(f"Single response readability score: {self.get_readability_score(instance['response'])}")
            print(f"Single response dependency graph depth: {self.get_dependency_graph_depth(instance['response'])}")

            print(f"Decomposed readability score: {np.mean(decomposed_readability_scores)}")
            print(f"Decomposed dependency graph depth: {np.mean(decomposed_dependency_graph_depths)}")



def main():

    parser = ArgumentParser()
    parser.add_argument("--input_fp", "-i", type=str, default=f"{PACKAGE_DIR}/assets/olmo_human_eval_responses.jsonl")
    parser.add_argument("--gpt_responses_fp", type=str, default=f"{PACKAGE_DIR}/assets/gpt_human_eval_responses.jsonl")
    parser.add_argument("--speech_reward_model_path", "-rm", type=str, default=f"{PACKAGE_DIR}/assets/gptj_reward_model")
    parser.add_argument("--prepare_hakiki_data", "-p", action="store_true")
    parser.add_argument("--overwrite", "-o", action="store_true")
    args = parser.parse_args()

    with open(args.input_fp, "r") as f:
        lines = f.readlines()
        eval_data = [json.loads(line) for line in lines]

    if os.path.exists(args.gpt_responses_fp):
        with open(args.gpt_responses_fp, "r") as f:
            lines = f.readlines()
            gpt_responses = [json.loads(line) for line in lines]

        eval_data.extend(gpt_responses)

    df = pd.DataFrame(eval_data)

    evaluator = SpeechLLMEval(speech_reward_model_path=args.speech_reward_model_path)
    # evaluator = SpeechLLMEval(speech_reward_model_path=None)

    # config name is the concatenation of the source and system prompt
    # if system_prompt is null, set it to "" 
    df["system_prompt"] = df["system_prompt"].fillna("")
    # remove /home/ubuntu/project/LLaMA-Factory/saves/ from source name 
    df["source"] = df["source"].apply(lambda x: x.replace("/home/ubuntu/project/LLaMA-Factory/saves/", ""))

    def cleanup_olmo_config(config): 
        config = config.replace("allenai/OLMo-7B-Instruct-hf", "")
        config = config.replace("olmo-7b-instruct/lora/", "")

        if "checkpoint" in config: 
            config = config.split("checkpoint")[0]
        config = config.replace("/", "")

        return config

    if "olmo" in args.input_fp.lower(): 
        df["source"] = df["source"].apply(lambda x: cleanup_olmo_config(x))

    def cleanup_falcon_config(config): 

        config = config.replace("tiiuae/falcon-7b-instruct", "")
        config = config.replace("falcon-7b-instruct/lora", "")

        if "checkpoint" in config: 
            config = config.split("checkpoint")[0]

        config = config.replace("dpo1", "dpo")
        config = config.replace("/", "")
        return config

    if "falcon" in args.input_fp.lower():
        df["source"] = df["source"].apply(lambda x: cleanup_falcon_config(x))

    df["config_name"] = df["source"] + df["system_prompt"] 

    ### compute stats 
    # filter for only factually correct ones
    if "correctness" in df.keys(): 
        # only keep those rows where all models got 5s for correctness to control for difference in model quality 
        df = df[df["correctness"] == "5"]
        n_model_types = df["model_type"].nunique()


        grouped_by_prompts = df.groupby("prompt").agg("count")
        prompts_with_all_responses = grouped_by_prompts[grouped_by_prompts["relevance"] == n_model_types].index.tolist()
        # keep only where the number of responses is equal to the number of models
        all_correct_df = df[df["prompt"].isin(prompts_with_all_responses)]



        all_correct_agg = all_correct_df.groupby("model_type")[["read.", "depend.", "wc", "voice_rm", "nonvocal"]].agg(["mean", "sem", "count"])

        # get correlation among scores & automatic metrics
        correlation_matrix = df[["correctness", "helpfulness", "informativeness", "length", "understandability", "relevance", "read.", "depend.", "wc", "voice_rm", "nonvocal"]].corr()
        all_correct_correlation_matrix = all_correct_df[["correctness", "helpfulness", "informativeness", "length", "understandability", "relevance", "read.", "depend.", "wc", "voice_rm"]].corr()

        print(correlation_matrix)

    responses = df["response"].tolist()
    prompts = df["prompt"].tolist()
    scores = evaluator.batch_evaluate(prompts, responses)

    for k,v in scores.items():
        df[k] = v

    # aggregate for all except on prompt and response
    agg = df.groupby("config_name")[["wc", "read.", "depend.", "voice_rm", "nonvocal"]].agg(["mean", "median", "sem"])

    configs_of_interest = [
        "base",
        "detailed",
        "icl",
        "dpobase",
        "dpo-detaileddetailed",
        "dpo-iclicl",
        "ppo-basebase",
        "ppo-detaileddetailed",
        "ppo-iclicl",
        "openaibase",
        "openaidetailed",
        "openaiicl",
        "reference"
    ]

    # filter for only configs of interest
    agg = agg[agg.index.isin(configs_of_interest)]
    # order the index based on the order of configs of interest
    agg = agg.reindex(configs_of_interest)

    # round to 2 decimal places
    agg = agg.round(2)
    print(agg)

    # print for table in overleaf 
    # output should like X_{y} where X is the mean and y is the standard error
    for idx, row in agg.iterrows():
        print(f"{idx} & {row['wc']['mean']}_{{ {row['wc']['sem']} }} & {row['read.']['mean']}_{{ {row['read.']['sem']} }} & {row['depend.']['mean']}_{{ {row['depend.']['sem']} }} & {row['voice_rm']['mean']}_{{ {row['voice_rm']['sem']} }} & {row['nonvocal']['mean']}_{{ {row['nonvocal']['sem']} }} \\")

    # save agg results as csv 
    agg.to_csv(args.input_fp.replace(".jsonl", "_agg_results.csv"))


    ### Configure match ups, all combinations of two between configs
    match_up_list = [(config1, config2) for config1 in df["config_name"].unique() for config2 in df["config_name"].unique() if config1 != config2]


    # match_up_list = list(set([tuple(sorted(match_up)) for match_up in match_up_list]))

    ### Compute head to head results based on reward model scores
    for match_up in match_up_list:
        model0 = match_up[0]
        model1 = match_up[1]

        model0_df = df[df["config_name"]==model0]
        model1_df = df[df["config_name"]==model1]

        # drop duplicates in prompt 
        model0_df = model0_df.drop_duplicates(subset=["prompt"], keep="first")
        model1_df = model1_df.drop_duplicates(subset=["prompt"], keep="first")

        # merge on prompt to c
        match_up_df = model0_df.merge(model1_df, on="prompt", suffixes=(f"_{model0}", f"_{model1}"))

        # compare scores on voice_suitalibity & read. 
        # metrics_of_interest = ["voice_rm", "read."]
        metrics_of_interest = ["voice_rm"]
 
        for metric in metrics_of_interest:
            model0_vs_model1_results = {model0:0, model1:0} 
            for idx, row in match_up_df.iterrows(): 
                
                model0_score = row[f"{metric}_{model0}"]
                model1_score = row[f"{metric}_{model1}"]

                if model0_score > model1_score: 
                    model0_vs_model1_results[model0] = model0_vs_model1_results.get(model0, 0) + 1
                elif model1_score > model0_score:
                    model0_vs_model1_results[model1] = model0_vs_model1_results.get(model1, 0) + 1
                else: 
                    model0_vs_model1_results["tie"] = model0_vs_model1_results.get("tie", 0) + 1
                
            # get percentages
            total = sum(model0_vs_model1_results.values())
            model0_vs_model1_results = {k: f"{v/(total+1e-6)*100:.2f}%" for k,v in model0_vs_model1_results.items()}
            # logger.info(f"{model0} vs {model1} results for {metric} for {total} samples:")
            model0_vs_model1_results["count"] = total 
            # pprint(model0_vs_model1_results)

    if not args.prepare_hakiki_data:
        return 
    

    ### Configure match ups 
    match_up_for_human_eval_list = [
        ("base", "detailed"),
        ("base", "icl"),
        ("base", "dpobase"), 
        ("base", "dpo-detaileddetailed"),
        ("base", "dpo-iclicl"),
        ("dpo-detaileddetailed", "detailed"),
        ("dpo-iclicl", "icl"),
        ("dpo-iclicl", "reference"),
        ("base", "ppo-basebase"),
        ("base", "ppo-detaileddetailed"),
        ("base", "ppo-iclicl"),
        ("ppo-detaileddetailed", "detailed"),
        ("ppo-detaileddetailed", "icl"),
        ("ppo-iclicl", "icl"),
        ("ppo-iclicl", "reference"),
        ("dpo-iclicl", "ppo-iclicl"),
        ("openaibase", "openaidetailed"), 
        ("openaibase", "openaiicl"),
        ("dpo-iclicl", "openaiicl"),
        ("icl", "openaiicl"),
    ]

    ### Create comparison data for human evaluation in format for hakiki 
    final_output_fn = args.input_fp.replace(".jsonl", "_hakiki.jsonl")

    # If there is prior data, then load it first and continue from there 
    if os.path.exists(final_output_fn) and not args.overwrite:
        with open(final_output_fn, "r") as f:
            lines = f.readlines()
            evaluation_data = [json.loads(line) for line in lines]
            for d in evaluation_data:
                if isinstance(d["data"], str):
                    d["data"] = json.loads(d["data"])

        # in case data was not written properly with the right format of having "data" wrapped around it for hakiki template processing
        fixed_data = [] 
        for d in evaluation_data: 
            
            if "data" not in d: 
                d = {"data": d}

            if "hash_key" not in d["data"]:
                model_name_order = sorted([d["data"]["model_0"], d["data"]["model_1"]])
                hash_text = f"{d['data']['instruction']}_{'_'.join(model_name_order)}"
                hash_key = hashlib.sha1(hash_text.encode('utf-8')).hexdigest()
                d["data"]["hash_key"] = hash_key

            fixed_data.append(d)
            
        with open(final_output_fn, "w") as f:
            for d in fixed_data:
                f.write(json.dumps(d) + "\n")
        evaluation_data = fixed_data

        hash_keys = [d["data"]["hash_key"] for d in evaluation_data]
        
    else: 
        evaluation_data = []
        hash_keys = [] 


    ### filter out rows where prompts are not suitable for voice, in case legacy test set was used 
    # use filter_voice_suitable_prompt_by_keywords
    df = df[df["prompt"].apply(lambda x: is_voice_suitable_prompt_based_on_keywords(x))]


    ### Create audio files for prompts
    logger.info("Creating audio files for user prompts")
    prompts = set(df["prompt"].tolist())
    prompts_audio_urls = {} 
    for prompt in tqdm(prompts): 
        audio_url = add_audio_for_text(prompt)
        prompts_audio_urls[prompt] = audio_url

    df["prompt_audio_url"] = df["prompt"].apply(lambda x: prompts_audio_urls[x])


    with open(final_output_fn, "a") as f:
        for match_up in match_up_for_human_eval_list: 
            skipped = 0 
            count = 0 

            model0 = match_up[0]
            model1 = match_up[1]

            model0_df = df[df["config_name"]==model0]
            model1_df = df[df["config_name"]==model1]

            # drop duplicates in prompt 
            model0_df = model0_df.drop_duplicates(subset=["prompt"], keep="first")
            model1_df = model1_df.drop_duplicates(subset=["prompt"], keep="first")

            # merge on prompt to c
            match_up_df = model0_df.merge(model1_df, on="prompt", suffixes=(f"_{model0}", f"_{model1}"))

            for idx, row in tqdm(match_up_df.iterrows(), total=len(match_up_df)): 
                prompt = row["prompt"]

                model_name_order = sorted([model0, model1])
                hash_text = f"{prompt}_{'_'.join(model_name_order)}"
                hash_key = hashlib.sha1(hash_text.encode('utf-8')).hexdigest()
                if hash_key in hash_keys:
                    continue

                randomly_selected_model0 = model0 if np.random.random() < 0.5 else model1
                randomly_selected_model1 = model1 if randomly_selected_model0 == model0 else model0    
                
                model0_response = row[f"response_{randomly_selected_model0}"]
                model1_response = row[f"response_{randomly_selected_model1}"]

                model0_audio_url = add_audio_for_text(model0_response)
                if model0_audio_url is None:
                    skipped += 1 
                    continue
                model1_audio_url = add_audio_for_text(model1_response) 
                if model1_audio_url is None:
                    skipped += 1 
                    continue

                comparison_sample = {
                    "instruction": cleanse_for_json(prompt),
                    "instruction_audio_url": prompts_audio_urls[prompt],
                    "model_0": randomly_selected_model0,
                    "model_1": randomly_selected_model1,
                    "model_0_response": cleanse_for_json(model0_response),
                    "model_1_response": cleanse_for_json(model1_response),
                    "model_0_audio_url": model0_audio_url,
                    "model_1_audio_url": model1_audio_url,
                    "set_type": "audio"
                }

                evaluation_data.append(comparison_sample)
                data = {
                   "data": json.dumps(comparison_sample)
                }   
                f.write(json.dumps(data) + "\n")

                count +=1 
                if count == 30: 
                    break

            print(f"Skipped {skipped} comparisons for {model0} vs {model1} because they were missing audio.")

        
if __name__ == "__main__":
    main()