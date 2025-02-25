from abc import ABC, abstractmethod
import requests
import transformers 
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer, FalconForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch 
import os
from typing import List, Dict 
from loguru import logger
from tqdm import tqdm
import math 
import numpy as np

from speechllm.modeling.reward_model.reward_model import GPTJRewardModel
from speechllm.utils import generate_openai_response, format_input_string_for_reward_model, PACKAGE_DIR


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            # breakpoint() 
            if torch.all((stop == input_ids[0])).item():
                return True

        return False


class BaseGenerativeModel(ABC): 

    def __init__(self): 
        pass

    @abstractmethod
    def generate_response(self, conversation_history):
        pass


class GenerativeModelCustomAPI(ABC):
    
    def __init__(self, api_endpoint): 
        self.api_endpoint = api_endpoint

    @abstractmethod
    def generate_response_single(self, conversation_history, **kwargs):
        pass 

    @abstractmethod
    def generate_response_batch(self, conversation_history, **kwargs):
        pass 


class OpenAIGenerativeModel(BaseGenerativeModel): 

    def __init__(self, model:str = "gpt-4o-2024-05-13"):
        self.name = model
        self.model = model

    def generate_multiple_responses(self, conversation_histories, **kwargs):
        responses = []
        for conversation_history in tqdm(conversation_histories): 
            responses.append(self.generate_response(conversation_history, **kwargs))
        return responses

    def generate_response(self, conversation_history, **kwargs):
        return generate_openai_response(conversation_history, model=self.model, **kwargs)

class FalconGenerativeModel(BaseGenerativeModel):

    def __init__(self, from_hf_directly:bool=True, model_path = None, model_size:str = "7b"): 

        self.USER_NAME = "User"
        self.BOT_NAME = "Assistant"
        self.hf_model_name = f"tiiuae/falcon-{model_size}-instruct" 
        self.name = f"falcon-{model_size}-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)

        if from_hf_directly:

            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.hf_model_name,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            ) 
        else: 
            assert model_path is not None, "Must provide model path if not loading from HF directly"
            self.model = FalconForCausalLM.from_pretrained(model_path, device_map="auto") 
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )


        stop_words = [self.USER_NAME, self.BOT_NAME, self.tokenizer.eos_token]
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
        self.stopping_criteria_list = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])


    def form_input_prompt(self, conversation_history): 
        """Reference: https://huggingface.co/spaces/tiiuae/falcon-chat/blob/main/app.py"""
        
        default_falcon_prompt = "The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Falcon, and a human user, called User. In the following interactions, User and Falcon will converse in natural language, and Falcon will answer User’s questions. Falcon was built to be respectful, polite and inclusive. Falcon was built by the Technology Innovation Institute in Abu Dhabi. Falcon will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins."


        assert conversation_history[0]["role"] == "system"
        assert conversation_history[1]["role"] == "user"
        assert conversation_history[-1]["role"] != "assistant"

        instruction = conversation_history[0]["content"]

        prompt = instruction
        for idx, turn in enumerate(conversation_history[1:]):
            if idx % 2 == 0:  
                assert turn["role"] == "user"
                prompt += f"\n{self.USER_NAME}: {turn['content']}"
            else: 
                assert turn["role"] == "assistant"
                prompt += f"\n{self.BOT_NAME}: {turn['content']}"

        prompt += f"\n{self.BOT_NAME}:"

        logger.debug(f"input prompt: {prompt}")
        return prompt

    def generate_response(self, conversation_history, top_k=10, num_return_sequences=1, max_length=2048, **kwargs):

        input_prompt = self.form_input_prompt(conversation_history)

        sequences = self.pipeline(
            input_prompt,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            max_length=max_length,
            # stopping_criteria=self.stopping_criteria_list,
            **kwargs
        )

        for seq in sequences: 
            index = seq["generated_text"].index(input_prompt)
            seq["generated_text"] = seq["generated_text"][index+len(input_prompt):].strip() 

        return sequences    

def create_reward_fn(reward_model_path: str = f"{PACKAGE_DIR}/assets/gptj_reward_model"):  # noqa:  C901
    reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"
    triton_host = os.environ.get("TRITON_HOST")

    # not used in our case. 
    if triton_host:

        import tritonclient.grpc as client_util 
        from tritonclient.utils import np_to_triton_dtype

        def prepare_tensor(name: str, input):
            t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
            t.set_data_from_numpy(input)
            return t

        triton_url, triton_model = triton_host.split("/")
        client = client_util.InferenceServerClient(url=triton_url, verbose=False)

        def reward_fn(samples, prompts, outputs):
            samples = [s + reward_tokenizer.eos_token for s in samples]
            input = reward_tokenizer(samples, padding=True, max_length=1024)

            mbs = 24
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = np.array(input.input_ids[batch_ixs], dtype=np.int32)

                result = client.infer(triton_model, [prepare_tensor("input_ids", input_ids)])
                rewards = result.as_numpy("rewards")
                out.extend(rewards)

            return out

    elif os.environ.get("RANK", "0") == "0":


        config = AutoConfig.from_pretrained("EleutherAI/gpt-j-6B")
        reward_model = GPTJRewardModel.from_pretrained(reward_model_path, config=config)

        reward_model.eval()
        reward_model.requires_grad_(False)

        reward_device = torch.cuda.device_count() - 1 if torch.cuda.is_available() else "cpu"
        
        reward_model = reward_model.half().to(reward_device)
        reward_batch_size = 24
        delta_reward = False

        def get_reward(samples):
            input = reward_tokenizer(
                samples,
                padding=True,
                truncation=True,
                max_length=reward_tokenizer.max_len_single_sentence,
                return_tensors="pt",
            ).to(reward_device)

            mbs = reward_batch_size
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = input.input_ids[batch_ixs]
                attention_masks = input.attention_mask[batch_ixs]
                rewards = reward_model(input_ids, attention_masks)
                out.extend(rewards.logits)
            return torch.hstack(out)

        def reward_fn(samples, prompts=None, **kwargs):
            samples = [format_input_string_for_reward_model(sample) for sample in samples]
            samples = [f"{s} {reward_tokenizer.eos_token}" for s in samples]
            rewards = get_reward(samples)

            return rewards

    else:
        reward_fn = True

    return reward_fn

