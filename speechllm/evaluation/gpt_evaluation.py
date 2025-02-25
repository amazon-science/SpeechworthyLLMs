from speechllm.models import OpenAIGenerativeModel
from speechllm.utils import form_openai_input_messages, PACKAGE_DIR
from speechllm.evaluation.claude_templates import FACTUALTIY_INPUT_TEMPLATE, FACTUALTIY_REST_OF_EVAL_TEMPLATE
from argparse import ArgumentParser
import pandas as pd
import json 
from json import JSONDecodeError
from pprint import pprint
from tqdm import tqdm 

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt-4o-2024-05-13")
parser.add_argument("--input_path", type=str, default=f"{PACKAGE_DIR}/assets/olmo_human_eval_responses.jsonl")

args = parser.parse_args()

with open(args.input_path, "r") as f:
    data = [json.loads(line) for line in f.readlines()]


openai_model = OpenAIGenerativeModel(model=args.model_name)


for d in tqdm(data, total=len(data)): 
    # continue 
    if "correctness" in d and d["correctness"] is not None: 
        continue 

    prompt = d["prompt"]
    response = d["response"]

    kwargs = {
        "user_prompt": prompt,
        "response": response
    }

    input_text = FACTUALTIY_INPUT_TEMPLATE.format(**kwargs) + FACTUALTIY_REST_OF_EVAL_TEMPLATE

    input_message = form_openai_input_messages(system_prompt=None, user_request=input_text)

    try: 
        eval_result = openai_model.generate_response(input_message)
        eval_result = eval_result.replace("`", "").strip()
        if eval_result.startswith("json"):
            eval_result = eval_result[4:]
        response = json.loads(eval_result)
        
        d["correctness"] = response["correctness"]["score"]

    except Exception as e: 
        print(eval_result)
        print(e)
        continue 

    with open(args.input_path, "w") as f:
        for d in data: 
            f.write(json.dumps(d) + "\n")





prompt = "What is the nickname for Spokane, Washington?"
response1 = "Lilac City"
response2 = "Lil Apple"
response3 = "It's known as Lilac City but also Lil Apple."

prompt = "What do you like?"
response1 = "Soccer"
response2 = "Apple"
response3 = "Noodles"

kwargs = {
    "user_prompt": prompt,
    "response": response1
}

input_text = FACTUALTIY_INPUT_TEMPLATE.format(**kwargs) + FACTUALTIY_REST_OF_EVAL_TEMPLATE

for response in tqdm([response1, response2, response3]):
    kwargs = {
        "user_prompt": prompt,
        "response": response
    }

    input_text = FACTUALTIY_INPUT_TEMPLATE.format(**kwargs) + FACTUALTIY_REST_OF_EVAL_TEMPLATE
    input_message = form_openai_input_messages(system_prompt=None, user_request=input_text)

    eval_result = openai_model.generate_response(input_message)

    try: 
        response = json.loads(eval_result)
        pprint(response)

    except JSONDecodeError as e: 
        print(eval_result)
        print(e)


