from datasets import load_dataset
from loguru import logger
import json 
import numpy as np 
from tqdm import tqdm
import os 
from speechllm.utils import generate_openai_response
from speechllm.constants import SYSTEM_PROMPTS, IN_CONTEXT_EXAMPLES
import re 
import unidecode

def form_detailed_prompt(user_request:str=None, n_icl_examples:int = 0):

    detailed_prompt = SYSTEM_PROMPTS["detailed"]

    if n_icl_examples > 0: 
        detailed_prompt = f"{detailed_prompt} You will be shown a few examples. Generate a response to the user for the current dialogue as the assistant."
        for ice in IN_CONTEXT_EXAMPLES[:n_icl_examples]:
            detailed_prompt += f"\n\n# Example dialogue\nUser: {ice[0]}\nAssistant: {ice[1]}"
        detailed_prompt += "\n\nNow directly respond to the user query without mentioning any of the examples."

        if user_request:
            detailed_prompt = f"{detailed_prompt}\n\n# Current dialogue\n User: {user_request}\nAssistant:"

    else: 
        if user_request:
            detailed_prompt = f"{detailed_prompt}\n\nUser: {user_request}\nAssistant:"

    return detailed_prompt

def form_detailed_prompt_for_api_llms(user_request:str, n_icl_examples:int = 0):

    detailed_system_prompt = SYSTEM_PROMPTS["detailed"]

    if n_icl_examples > 0: 
        detailed_system_prompt = f"{detailed_system_prompt} You will be shown a few examples. Generate a response to the user for the current dialogue as the assistant."
        for ice in IN_CONTEXT_EXAMPLES[:n_icl_examples]:
            detailed_system_prompt += f"\n\n# Example dialogue\nUser: {ice[0]}\nAssistant: {ice[1]}"


        user_message = f"# Current dialogue\nUser: {user_request}\nAssistant:"

    else: 
        user_message = f"User: {user_request}\nAssistant:"

    messages_in_openai_format = [
        {
            "role": "system",
            "content": detailed_system_prompt
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    return messages_in_openai_format

def keep_alphabet_only(text:str):
    # change special characters like ç to c, like mturk automatically does
    text = unidecode(text)

    # utility function for matching prompts that may contain non-alphabetic characters that gets processed differently by json 
    return re.sub(r'[^a-zA-Z ]+', '', text).strip()


def is_voice_suitable_prompt_based_on_keywords(user_instruction:str) -> bool: 
            # use basic keywords as initial filter
    keywords = [
        "write", 
        "python", 
        "text", 
        "given", 
        "https", 
        "extract",
        "based on",
        "summarize",
        "classify", 
        "mentioned",
        "which of these",
        "which one of these",
        "according to",
        "which of these",
        "determine",
        "list",
        "change this",
        "from the passage provided", 
        "select",
        "following", 
        "categorize", 
        "which is a",
        "choose",
        "pronounce", # you won't ask about pronunciation unless the users spell it out, play it safe to remove including these edge cases   
    ]

    for k in keywords: 
        if k in user_instruction.lower(): 
            return False
        
    return True 

def is_suitable_for_voice(user_instruction:str) -> bool: 
    """Use filter words and GPT-3.5 to filter out instructions unsuitable for voice. 
    """
    return is_voice_suitable_prompt_based_on_keywords(user_instruction) and is_voice_suitable_using_openai(user_instruction)


def is_voice_suitable_using_openai(user_instruction:str) -> bool: 
    # ask openai to check if the instruction is suitable for voice
    system_prompt = "Is the following request suitable for being asked to a voice assistant and be  answered by it? A request is not suitable if it contains content that cannot be vocalized or results in a response that cannot be easily vocalized, such as code or long-form text, that cannot be delivered at smaller chunks through multiple turns of dialogue. Answer with Yes or No and a brief reason why. \n\n"
    messages = [{"role":"user", "content": user_instruction}]

    response = generate_openai_response(system_prompt=system_prompt, messages = messages)
    logger.info(f"user_instruction: {user_instruction}")
    logger.info(f"openai response: {response}")

    if "yes" in response.lower(): 
        return True
    else: 
        return False 

class DollyDataset: 

    def __init__(self, args): 
        self.args = args 
        # https://huggingface.co/datasets/databricks/databricks-dolly-15k
        self.dataset = load_dataset("databricks/databricks-dolly-15k")
        self.sequential = [train_sample for train_sample in self.dataset["train"]]
        self.long_response_threshold = args.long_response_threshold



    def is_long_response(self, response:str) -> bool: 
        return len(response.split()) > self.long_response_threshold

    def count_long_responses(self):
        # print length statistics of the responses

        lens = [len(sample["response"].split()) for sample in self.sequential]
        logger.info(f"Original response length statistics:\n\tmean: {np.mean(lens)}\n\tstd: {np.std(lens)}\n\tmax: {np.max(lens)}\n\tmin: {np.min(lens)}")

        long_response = [sample for sample in self.sequential if self.is_long_response(sample["response"])]
        logger.info(f"number of long responses (> {self.long_response_threshold}): {len(long_response)}")

        return len(long_response)

    def collect_voice_prompts(self): 

        voice_suitable_samples = []
        
        n_total_candidates = len(self.sequential)
        logger.info(f"total number of candidates from Dolly 15k: {n_total_candidates}")

        if os.path.exists(self.args.output_fn):
            logger.info(f"output file already exists: {self.args.output_fn}")
            voice_suitable_samples = [json.loads(line) for line in open(self.args.output_fn, "r")]

        else: 
            for sample in tqdm(self.sequential, total=len(self.sequential)): 

                user_instruction = sample["instruction"]
                response = sample["response"]
                # if not self.is_long_response(response): 
                #     continue 

                if self.is_suitable_for_voice(user_instruction): 
                    logger.debug(f"adding sample: {user_instruction}")
                    voice_suitable_samples.append(sample)
                else: 
                    logger.debug(f"not adding sample: {user_instruction}")


            with open(self.args.output_fn, "w") as f: 
                for sample in voice_suitable_samples: 
                    f.write(json.dumps(sample))
                    f.write("\n")        

        logger.info(f"total number of selected samples: {len(voice_suitable_samples)}")
            
        self.voice_suitable_samples = voice_suitable_samples

    def print_statistics(self): 
        """Print statistics about the dataset.
        """

        # print average number of words in instruction 
        lens = [len(sample["instruction"].split()) for sample in self.voice_suitable_samples]
        logger.info(f"Voice suitable instruction length statistics:\n\tmedian: {np.median(lens)}\n\tstd: {np.std(lens)}\n\tmax: {np.max(lens)}\n\tmin: {np.min(lens)}")

        # print average number of words in response
        lens = [len(sample["response"].split()) for sample in self.voice_suitable_samples]
        logger.info(f"Voice suitable response length statistics:\n\tmedian: {np.median(lens)}\n\tstd: {np.std(lens)}\n\tmax: {np.max(lens)}\n\tmin: {np.min(lens)}")

        # print average number of turns and words in decomposed responses 
        if "decomposed-gpt-4-response" in self.voice_suitable_samples[0].keys():
            turns = [len(sample["decomposed-gpt-4-response"][1::2]) for sample in self.voice_suitable_samples if "decomposed-gpt-4-response" in sample.keys()] 

            logger.info(f"Voice suitable decomposed response turn statistics:\n\tmedian: {np.median(turns)}\n\tstd: {np.std(turns)}\n\tmax: {np.max(turns)}\n\tmin: {np.min(turns)}")

            all_lens = [len(turn['text'].split()) for sample in self.voice_suitable_samples if "decomposed-gpt-4-response" in sample.keys() for turn in sample["decomposed-gpt-4-response"][1::2]]
            logger.info(f"Voice suitable decomposed response length statistics:\n\tmedian: {np.median(all_lens)}\n\tstd: {np.std(all_lens)}\n\tmax: {np.max(all_lens)}\n\tmin: {np.min(all_lens)}")

        return 

