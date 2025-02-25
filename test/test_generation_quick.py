from argparse import ArgumentParser
from loguru import logger
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6B")
parser.add_argument("--lora_model", action="store_true")
args = parser.parse_args()

model_name = args.model_name


# for quickly testing whether checkpoints generate reasonable outputs 

if args.lora_model:
    model = AutoPeftModelForCausalLM.from_pretrained(model_name)
else: 
    model = AutoModelForCausalLM.from_pretrained(model_name)

model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(input_string): 

    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=100, do_sample=True, temperature=0.9, top_k=50, top_p=0.95, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

while True: 
    user_input = input("User: ")
    response = generate_response(user_input)
    print(f">>> Assistant: {response}")
