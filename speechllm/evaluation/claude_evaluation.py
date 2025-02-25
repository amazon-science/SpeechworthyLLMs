# shared by Kexuan and Karishma

import os, json, time
# from langchain.llms.bedrock import Bedrock
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from speechllm.evaluation.claude_templates import REST_OF_EVAL_TEMPLATE, INPUT_TEMPLATE
from speechllm.utils import PACKAGE_DIR
import json
from json import JSONDecodeError
from pprint import pprint
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import traceback
import textstat
from argparse import ArgumentParser


# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Helper utilities for working with Amazon Bedrock from Python notebooks"""
# Python Built-Ins:
import os
from typing import Optional

# External Dependencies:
import boto3
from botocore.config import Config


def get_bedrock_client(
    assumed_role: Optional[str] = None,
    profile_name: Optional[str] = None,
    target_region: Optional[str] = None,
    service_name: str = "bedrock-runtime",
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    endpoint_url :
        Optional override for the Bedrock service API Endpoint. If setting this, it should usually
        include the protocol i.e. "https://..."
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    """
    if target_region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = target_region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


def createAgent(max_tokens=1024, temperature=0.2, top_k=250, top_p=1, stop_sequences=[]):
    """
    bedrock-runtime – Contains runtime plane APIs for making inference requests for models hosted in Amazon Bedrock
    bedrock – Contains control plane APIs for managing, training, and deploying models
    """
    load_dotenv()
    boto3_bedrock_client = get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        profile_name=os.environ.get("AWS_PROFILE", None),
        target_region=os.environ.get("AWS_REGION", None),
        service_name="bedrock-runtime",
    )

    inference_modifier = {
        'max_tokens': max_tokens,
        "temperature": temperature,  # Temperature (temperature)– Use a lower value to decrease randomness in the response.
        "top_k": top_k,  # Top K (topK) – Specify the number of token choices the model uses to generate the next token.
        "top_p":top_p,  # Top P (topP) – Use a lower value to ignore less probable options.
        "stop_sequences": stop_sequences
    }
    llmAgent = ChatBedrock(
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0",
        client = boto3_bedrock_client, 
        model_kwargs = inference_modifier 
    )
    return llmAgent


def callAgent(llmAgent, prompt):

    num_tokens = llmAgent.get_num_tokens(prompt)
    print(f"Our prompt has {num_tokens} tokens")

    messages = [
        HumanMessage(
            content=prompt
        )
    ]

    # call agent
    start_time = time.time()
    response = llmAgent.invoke(messages)  # or textgen_llm.invoke(input=prompt)
    print("Time taken: %s secs." % (time.time() - start_time))
    return response


def example(): 
    # Define the template
    template = "Hello, my name is {name}. I am a {profession}. I am {age} years old. I live in {city}."
    # Define the input variables
    kwargs = {
        "name": "Alice",
        "profession": "Data Scientist",
        "age": "30",
        "city": "New York",
    }

    # Create the agent
    llmAgent = createAgent()

    # Call the agent
    response = callAgent(llmAgent, template, kwargs)
    print(response)
    return 

def consistency_test(_print=False): 

    llmAgent = createAgent()

    kwargs = {
        "user_prompt": "What are the most common languages in the world?",
        "response": "The most common languages in the world by number of speakers are:\n\n1. Spanish - around 460 million speakers\n2. English - around 1 billion speakers\n3. Chinese - around 1.3 billion speakers\n4. Spanish (in the United States) - around 66 million speakers\n5. English (in the world) - around genetically speaking, we have only 46 couch potatoes code lines in our DNA.\n\nIt's important to note these numbers are\u53e3hladxiety and can change depending on factors such as politics, territories and standardization",
    }

    input_text = INPUT_TEMPLATE.format(**kwargs) + REST_OF_EVAL_TEMPLATE

    results = [] 

    for _ in tqdm(range(10)): 
        response = callAgent(llmAgent, input_text)

        if _print: 
            pprint(response.additional_kwargs)

        try: 
            response_json = json.loads(response.content)
            if _print: 
                pprint(response_json)

            
            results.append(response_json)

        except JSONDecodeError as e: 
            print(e)
            print(response.content)
            
    # extract scores only 
    results_df_data = [{k: int(v['score']) for k, v in result.items()} for result in results]
    
    results_df = pd.DataFrame(results_df_data)

    #convert column values to int
    results_df = results_df.astype(int)

    # get mean and std
    print(results_df.mean())
    print(results_df.std())
    print(results_df.sem())

def process_json_output(response): 

    response = response.replace("```", "")
    response = response.replace("json", "") if response.startswith("json") else response
    response = response.strip() 

    return response

def main():

    llmAgent = createAgent()

    parser = ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default=f"{PACKAGE_DIR}/assets/human_eval_responses.jsonl")
    args = parser.parse_args()

    path = args.path

    with open(path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
        # data = data[:1]

    results = [] 
    for idx, d in tqdm(enumerate(data), total=len(data)): 
        prompt = d.pop("prompt")
        for model_type, response in d.items(): 

            kwargs = {
                "user_prompt": prompt,
                "response": response
            }

            input_text = INPUT_TEMPLATE.format(**kwargs) + REST_OF_EVAL_TEMPLATE

            eval_result = callAgent(llmAgent, input_text)

            try:
                response_json = json.loads(process_json_output(eval_result.content))

                # keep only the scores and change scores to ints
                response_json = {k: int(v['score']) for k,v in response_json.items()}

                response_json["model_type"] = model_type
                response_json["response"] = response
                response_json["prompt"] = prompt
                results.append(response_json)
            except Exception as e:
                #print trace 
                traceback.print_exc()
                print(f"Error: {e}")
                print(eval_result.content)

        if idx+1 % 10 == 0: 
            results_df = pd.DataFrame(results)
            # save results as jsonl
            save_path = path.replace(".jsonl", "_with_scores.jsonl")
            with open(save_path, "w") as f:
                for line in results:
                    f.write(json.dumps(line) + "\n")



    results_df = pd.DataFrame(results)
    # save results as jsonl
    save_path = path.replace(".jsonl", "_with_model_scores.jsonl")
    with open(save_path, "w") as f:
        for line in results:
            f.write(json.dumps(line) + "\n")

    
    # aggregate by model_type 
    results_df_agg = results_df.groupby("model_type")[["relevance", "helpfulness", "correctness", "understandability", "informativeness", "length"]].agg(["mean", "std", "sem"])

    print(results_df_agg)


if __name__ == "__main__":
    # example() 
    main() 