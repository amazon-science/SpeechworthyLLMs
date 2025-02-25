# adds decomposed multi-turn conversations from a dataset with long responses.

import json 
from argparse import ArgumentParser
from speechllm.utils import generate_openai_response_to_user_request, add_audio_for_multiturn
from tqdm import tqdm 
from loguru import logger 

parser = ArgumentParser()
parser.add_argument("--input", type=str, help="path to input file")
parser.add_argument("--output", type=str, default="", help="path to output file")
parser.add_argument("--min_word_length", type=int, default=50, help="minimum word length for original response")
parser.add_argument("--add_audio", action="store_true", help="whether to add audio for the decomposed response")
parser.add_argument("--model", type=str, default="gpt-4", help="which model to use for generating decomposed response")
parser.add_argument("--response-key", type=str, default="gpt-4-response", help="which key to use for the response to decompose")
parser.add_argument("--exact-match", action="store_true", help="whether to only keep samples where the first question is exactly the same as the original question")
args = parser.parse_args()

if args.output =="": 
    args.output = args.input 

MIN_WORD_LENGTH = args.min_word_length


def parse_decomposed_multiturn(decomposed_response): 
    turns = decomposed_response.split("\n")
    turns = [turn for turn in turns if turn.strip()]  

    # check that they are in correct format 
    # odd turns start with Q: and does not contain A: 
    try: 
        assert all([turn.startswith("Q:") and "A:" not in turn for idx, turn in enumerate(turns) if idx % 2 == 0])
        # even turns start with A: and does not contain Q:
        assert all([turn.startswith("A:") and "Q:" not in turn for idx, turn in enumerate(turns) if idx % 2 == 1])
    except Exception as e:
        # breakpoint() 
        print(turns)
        return False 

    # remove A: Q: formatting 
    turns = [turn.replace("Q:", "").replace("A:", "").strip() for turn in turns]

    return turns 


with open(args.input, "r") as f: 
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

for sample in tqdm(data): 
    original_response = sample[args.response_key]

    decomposed_result_key = f"decomposed-{args.model}-response"

    if decomposed_result_key in sample and isinstance(sample[decomposed_result_key], str):            
        sample.pop(decomposed_result_key)

    # remove samples that have decomposed conversations with first question that are not the same as the original question
    if args.exact_match and decomposed_result_key in sample and sample[decomposed_result_key][0] != sample['instruction']:
        sample.pop(decomposed_result_key)

    if len(original_response.split()) < MIN_WORD_LENGTH: 
        continue 

    if decomposed_result_key not in sample:

        instruction = sample["instruction"]
        prompt = f"Question: {instruction}\nResponse: {original_response}\n\nBreak down this question and response to a conversation with multiple subquestions and answers using Q: x\nA: y\n format. Make Q and A take one turn each. Make sure that the first question in the conversation is exactly the same as the original question and maintain the same point of view throughout all questions.\n\n" 

        new_response = generate_openai_response_to_user_request(user_request=prompt, model=args.model)

        turns_plain = parse_decomposed_multiturn(new_response)
        if turns_plain is False:
            continue

        if instruction != turns_plain[0]: 
            logger.warning(f"Original instruction: {instruction}")
            logger.warning(f"First turn: {turns_plain[0]}")
            # breakpoint() 

        turns = [{"text": turn} for turn in turns_plain]        
        if args.add_audio:
            turns = add_audio_for_multiturn(turns, override=True)

        sample[decomposed_result_key] = turns

        with open(args.output, "w") as f: 
            for sample in data: 
                f.write(json.dumps(sample) + "\n")


