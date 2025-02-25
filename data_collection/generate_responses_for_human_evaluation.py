# Description: This script is used to generate/extract responses for human/model evaluation.
# Usage shown in generate_response_script.sh

import json 
from speechllm.models import OpenAIGenerativeModel
from speechllm.constants import BASE_SYSTEM_PROMPT
from speechllm.utils import form_openai_input_messages, PACKAGE_DIR
from speechllm.dataset_utils import is_voice_suitable_prompt_based_on_keywords, form_detailed_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer 
import random 
import os 
from tqdm import tqdm 
from argparse import ArgumentParser
from loguru import logger
from peft import AutoPeftModelForCausalLM
import torch
from typing import List
import pandas as pd
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from training with llamafactory 
def format_falcon_prompt(system:str, query:str) -> str: 
    if system is None: 
        input_string = f"User: {query}\nFalcon:"
    else: 
        input_string = f"{system}\n\nUser: {query}\nFalcon:"

    return input_string

def format_olmo_prompt(system:str, query:str) -> str: 

    if system is None: 
        input_string = f"<|user|>\n{query} <|assistant|>\n"
    else: 
        input_string = f"{system} <|user|>\n{query} <|assistant|>\n"    

    return input_string

def get_hf_model_responses(input_strings: List[str], model, tokenizer, batch_size = 8):

    logger.info(f"Generating responses from model: {model.config.name_or_path}")

    model_responses = []

    # do batch generation 
    for i in tqdm(range(0, len(input_strings), batch_size)):
        batch_input_strings = input_strings[i:i+batch_size]
        batch_tokenized_input = tokenizer(batch_input_strings, return_tensors="pt", padding=True, truncation=True).to(device)
        batch_output = model.generate(
            input_ids = batch_tokenized_input.input_ids, # need to name for peftmodels
            max_new_tokens=512, num_return_sequences=1, 
            do_sample=True, 
            top_k=0,
            top_p=0.9, 
            temperature=0.6
        )
        batch_output = [tokenizer.decode(output, skip_special_tokens=True) for output in batch_output]
        model_responses.extend(batch_output)

    logger.info(f"Number of responses generated from model: {len(model_responses)}")
    
    only_model_responses = [] 
    for input_string, response in zip(input_strings, model_responses):
        only_model_response = response.replace(input_string, "").strip()

        # postprocessing for falcon 
        only_model_response = only_model_response.split("User")[0].strip()
        only_model_response = only_model_response.split("# Example")[0].strip()
        # post processing for olmo 
        only_model_response = only_model_response.split("<|assistant|>")[0].strip()

        only_model_responses.append(only_model_response)


    return only_model_responses



def main(): 
    HOME = os.environ["HOME"]

    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--openai_version", type=str, default="gpt-4o", help="One of [gpt-4o, gpt-4-turbo]")
    parser.add_argument("--not_lora", action="store_true", help="use if model is not finetuned with peft & lora")
    parser.add_argument("--system_prompt_type", type=str, default="base", help="One of [none, base, detailed, icl, all]")
    parser.add_argument("--test_path", type=str, default=f"{PACKAGE_DIR}/assets/databricks_dolly_voice_prompt_samples_full.jsonl")
    parser.add_argument("--save_dir", type=str, default=f"{PACKAGE_DIR}/assets/")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_size", type=int, default=4)
    parser.add_argument("--full_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.test_path, "r") as f:
        test_data = [json.loads(line) for line in f.readlines()]

    trust_remote_code = True if "falcon" not in args.model_name_or_path else False

    # load previous data if exists
    if args.save_path:
        save_path = args.save_path
    else: 
        save_path = os.path.join(args.save_dir, f"human_eval_responses.jsonl")

    if args.test:
        save_path = save_path.replace(".jsonl", "_test.jsonl")
        
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
            df = pd.DataFrame(data)
    else: 
        df = pd.DataFrame(columns=["prompt", "source", "system_prompt", "response", "input_string", ])


    # keep only the prompts that are suitable for voice and are ascii
    test_data = [item for item in test_data if is_voice_suitable_prompt_based_on_keywords(item["instruction"]) and item["instruction"].isascii()]

    # get prompts from test data
    shuffled_indices = list(range(len(test_data)))
    random.seed(args.seed)
    random.shuffle(shuffled_indices)
    shuffled_indices = shuffled_indices[:args.full_size] if not args.test else shuffled_indices[:args.test_size]

    samples = [test_data[i] for i in shuffled_indices]

    prompts = [d["instruction"] for d in samples]
    reference_responses = [d["response"] for d in samples] 

    # if prompt in df is not in prompts, remove them 
    df = df[df["prompt"].isin(prompts)]

    new_rows = []

    for prompt, response in zip(prompts, reference_responses):
        if df[(df["prompt"] == prompt) & (df["source"] == "reference") & (df["response"] == response)].shape[0] > 0:
            continue
        new_rows.append({"prompt": prompt, "response": response, "source": "reference", "system_prompt": None, "input_string": prompt, })

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    system_prompt_type = args.system_prompt_type
    # get system prompt to use 
    if system_prompt_type == "none":
        system_prompt = None
    elif system_prompt_type == "base":
        system_prompt = BASE_SYSTEM_PROMPT
    elif system_prompt_type == "detailed":
        system_prompt= form_detailed_prompt(n_icl_examples=0)
    elif system_prompt_type == "icl":
        system_prompt = form_detailed_prompt(n_icl_examples=5)
    else: 
        NotImplementedError(f"System prompt type `{args.system_prompt_type}` not supported")

    format_func_dict = {
        "falcon": format_falcon_prompt,
        "olmo": format_olmo_prompt,
        "openai": form_openai_input_messages,
    }

    for k, v in format_func_dict.items():
        if k in args.model_name_or_path.lower():
            format_func = v
            break

    input_strings_or_messages = [format_func(system_prompt, prompt) for prompt in prompts]

    # skip those that were already generated with the same input from before, or overwrite if overwrite flag is set
    unseen_input_strings_or_messages = [] 
    unseen_prompts = [] 

    for prompt, input_string_or_message in zip(prompts, input_strings_or_messages):
        if args.model_name_or_path == "openai":
            assert isinstance(input_string_or_message, list)
            input_string = " ".join([msg['content'] for msg in input_string_or_message])
        else: 
            input_string = input_string_or_message

        if df[(df["prompt"] == prompt) & (df["system_prompt"] == system_prompt_type) & (df["source"] == args.model_name_or_path) & (df["input_string"] == input_string)].shape[0] > 0:
            if args.overwrite: 
                df = df.drop(df[(df["prompt"] == prompt) & (df["system_prompt"] == system_prompt_type) & (df["source"] == args.model_name_or_path) & (df["input_string"] == input_string)].index)
            else: 
                continue
        unseen_input_strings_or_messages.append(input_string_or_message)
        unseen_prompts.append(prompt)

    logger.info(f"Number of unseen input strings: {len(unseen_prompts)}")
    logger.info(f"Number of seen input strings: {len(input_strings_or_messages) - len(unseen_input_strings_or_messages)}")

    if len(unseen_input_strings_or_messages) == 0:
        logger.info(f"No unseen input strings to generate responses for. Exiting.")
    else: 
        # initialize model 
        if args.model_name_or_path == "openai": 
            # TODO 
            model = OpenAIGenerativeModel(args.openai_version)
            get_model_response_func = model.generate_multiple_responses
            tokenizer = None
        elif args.model_name_or_path == "claude":
            # TODO 
            # model = ClaudeGenerativeModel()
            # format_func = form_claude_input_messages
            # get_model_response_func = model.generate_response
            tokenizer = None
            pass
        else: 
            if args.not_lora: 
                model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=trust_remote_code, torch_dtype=torch.float16)
            else: 
                model = AutoPeftModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=trust_remote_code, torch_dtype=torch.float16)

            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=trust_remote_code, padding_side="left")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            get_model_response_func = partial(get_hf_model_responses, batch_size=args.batch_size, tokenizer=tokenizer, model=model)

            model.to(device)

        # generate responses from model for unseen prompts
        model_responses = get_model_response_func(unseen_input_strings_or_messages)

        new_rows =[] 
        # add them to dataframe
        for prompt, response, input_string_or_message in zip(unseen_prompts, model_responses, unseen_input_strings_or_messages):
            if args.model_name_or_path == "openai":
                input_string = " ".join([msg['content'] for msg in input_string_or_message])
            else: 
                input_string = input_string_or_message
            new_rows.append({"prompt": prompt, "response": response, "source": args.model_name_or_path, "input_string": input_string, "system_prompt": system_prompt_type})
        
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        save_path = args.save_path if args.save_path else save_path
        if args.test:
            save_path = save_path.replace(".jsonl", "_test.jsonl")


        # load again before saving in case other files wrote to it 
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                existing_data = [json.loads(line) for line in f.readlines()]
                existing_df = pd.DataFrame(existing_data)

            # get jsonl form list from df 
            df = pd.concat([df, existing_df], ignore_index=True)
            # drop duplicates 
            df = df.drop_duplicates(subset=["prompt", "source", "system_prompt"])

        df.to_json(save_path, orient="records", lines=True)

    df = df.drop_duplicates(subset=["prompt", "source", "system_prompt"])

    df.to_json(save_path, orient="records", lines=True)

    # summary of counts of each source + system prompt type
    summary = df.groupby(["source", "system_prompt"]).size().reset_index(name="counts")
    logger.info(f"Summary of counts:")
    summary["source"] = summary["source"].apply(lambda x: x.replace("/home/ubuntu/project/LLaMA-Factory/saves", ""))
    print(summary)


if __name__ == "__main__":
    main()