import openai 
import os 
import boto3 
from loguru import logger
import time 
import traceback
import hashlib
from typing import List, Dict
from unidecode import unidecode 
import re
import base64
import requests
from pathlib import Path 

from .constants import BASE_SYSTEM_PROMPT, PREVIOUS_BASE_SYSTEM_PROMPT

openai.api_key = os.environ.get("OPENAI_API_KEY", "")
PACKAGE_DIR = Path(__file__).resolve().parent
HOME_DIR = Path.home()

def cleanse_for_json(text:str): 
    # remove any characters that cannot be parsed by JSON in javascript
    text = unidecode(text)
    # return text.replace('\n', ' ').replace('"', ''),
    text = text.replace('"', "'")
    return text

def generate_openai_response_to_user_request(user_request:str, model:str="gpt-3.5-turbo", **kwargs) -> None: 

    while True: 
        try: 
            completion = openai.ChatCompletion.create(
                model = model, 
                messages = [
                    {"role": "user", "content": user_request},
                ], 
                **kwargs
            )

            return completion.choices[0].message["content"]
        except Exception as e: 
            logger.error(e) 
            traceback.print_exc()
            time.sleep(1)
            continue 

def generate_openai_response(messages: List[Dict[str,str]], model:str="gpt-4o-2024-05-13", system_prompt:str = "", **kwargs) -> None: 

    if system_prompt: 
        messages = [{"role": "system", "content": system_prompt}]  + messages 

    while True: 
        try: 
            completion = openai.ChatCompletion.create(
                model = model, 
                messages = messages,
                **kwargs
            )

            return completion.choices[0].message["content"]
        except Exception as e: 
            logger.error(e) 
            traceback.print_exc()
            time.sleep(1)
            continue 


def create_audio_url(subkey, cloud_front_url="https://d2zzmiygasj3dp.cloudfront.net/"): 
    return os.path.join(cloud_front_url, subkey)
     
def add_audio_for_multiturn(multiturn, override=False): 

    instruction = multiturn[0]['text']
    unique_id = hashlib.sha1(instruction.encode('utf-8')).hexdigest()

    new_turns = [] 
    for idx, turn in enumerate(multiturn): 
        subkey = f"{unique_id}_{idx}.mp3"
        output_key = f'data/audio/{subkey}'
        output_bucket = 'jcho-voicellm'
        text = turn['text']
        audio_created = create_audio_file(text=text, output_bucket=output_bucket, output_key=output_key, overwrite=override)

        if audio_created:
            public_audio_url = create_audio_url(subkey)
        else: 
            public_audio_url = None
            
        new_turns.append(
            {"text": text, "audio_url": public_audio_url}
        )

    return new_turns 


def add_audio_for_text(text, overwrite=False, output_bucket = "jcho-voicellm", sample_key_to_overwrite:str = None): 
    unique_id = hashlib.sha1(text.encode('utf-8')).hexdigest()
    subkey = f"{unique_id}.mp3"
    output_key = f'data/audio/{subkey}'
    audio_created = create_audio_file(text=text, output_bucket=output_bucket, output_key=output_key, overwrite=overwrite)

    if audio_created:
        public_audio_url = create_audio_url(subkey)
    else:  
        public_audio_url = None

    return public_audio_url

def add_audio_for_sample(sample, overwrite=False, output_bucket = 'jcho-voicellm', sample_key_to_overwrite:str = None): 
    instruction = sample['instruction']
    unique_id = hashlib.sha1(instruction.encode('utf-8')).hexdigest()

    # create audio files of all text and add public url for each 
    keys = list(sample.keys())
    for key in keys: 
        
        if 'response' not in key and 'instruction' not in key: 
            continue 

        if 'audio' in key:  
            continue 

        subkey = f"{unique_id}_{key}.mp3"
        output_key = f'data/audio/{subkey}'
        
        text = sample[key]

        if not isinstance(text, str): 
            continue 

        if sample_key_to_overwrite and sample_key_to_overwrite in key:
            audio_created = create_audio_file(text=text, output_bucket=output_bucket, output_key=output_key, overwrite=True)
        else: 
            audio_created = create_audio_file(text=text, output_bucket=output_bucket, output_key=output_key, overwrite=overwrite)

        if audio_created: 
            public_audio_url = create_audio_url(subkey)
        else: 
            public_audio_url = None 

        sample[f"{key}_audio_url"] = public_audio_url

    return sample 

def create_audio_file(text:str, output_bucket:str, output_key:str, overwrite:bool) -> bool:

    # Create a session using your AWS credentials
    session = boto3.Session()
    s3 = session.client('s3')

    # check if item already exists in s3
    if not overwrite and s3.list_objects_v2(Bucket=output_bucket, Prefix=output_key).get('Contents'):
        logger.info(f'Audio file already exists in S3: {output_bucket}/{output_key}')
        return True

    # Create an instance of the Polly client
    polly = session.client('polly')
    
    # Specify the desired voice and output format
    voice_id = 'Ruth'  # Replace with the desired Polly voice ID
    output_format = 'mp3'  # You can also choose 'ogg_vorbis' or 'pcm' format
    
    # Make a call to Polly to synthesize speech
    try: 
        response = polly.synthesize_speech(
            OutputFormat=output_format,
            Text=text,
            VoiceId=voice_id,
            Engine= "generative"
        )
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return False 
    
    # Save the audio file to S3 bucket
    s3.put_object(
        Bucket=output_bucket,
        Key=output_key,
        Body=response['AudioStream'].read()
    )
    
    logger.info(f'Audio file created and stored in S3: {output_bucket}/{output_key}')
    return True 


def form_openai_input_messages(system_prompt, user_request): 

    if system_prompt is None: 
        system_prompt = "You are a helpful assistant."

    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]

    return conversation_history

def format_prompt_response_for_reward_model(prompt, response): 
    # prompt: user query text only 
    # response: response text only

    return f"User: {prompt}\nAssitant: {response}"


def format_input_string_for_reward_model(sample):
    '''
    Format input for reward model
    e.g. <s>[INST] You are a helpful, respectful and honest voice assistant.\n\nWhy do the Dutch wear Orange? [/INST] Because they like orange.
    shoud become
    User: Why do the Dutch wear Orange? Assistant: Because they like orange. 
    '''

    sample = sample.split(BASE_SYSTEM_PROMPT)[-1].strip()
    sample = sample.split(PREVIOUS_BASE_SYSTEM_PROMPT)[-1].strip()
    # add whitespace after User:
    sample = re.sub(r'User:', 'User: ', sample)

    sample = f"User: {sample}" if not sample.startswith("User:") else sample

    # for mistral & llama 
    sample = sample.replace("[INST]", "")
    sample = sample.replace("[/INST]", "\nAssistant: ")

    # for falcon 
    sample = sample.replace("<|im_end|>", "\nAssistant: ")
    sample = sample.replace("<|im_start|>assistant", "")

    # for olmo 
    sample = sample.replace("<|user|>\n", "")
    sample = sample.replace("<|assistant|>\n", "Assistant: ")
    sample = sample.replace("<|assistant|>", "Assistant: ")

    # replace multi whitespace with single whitespace
    sample = re.sub(r'\s+', ' ', sample)

    # add new line in front of Assistant if there isn't one
    if "Assistant:" in sample:
        split = sample.split("Assistant:")
        sample = f"{split[0].strip()}\nAssistant: {split[1].strip()}"
    
    return sample.strip()


def form_model_input(messages_openai_format, tokenizer, format_strategy:str="uniform_with_speech_prompt"): 

    if "apply_template" in format_strategy:

        if 'with_speech_prompt' in format_strategy:
            messages_openai_format[0]['content'] = BASE_SYSTEM_PROMPT + "\n\n" + messages_openai_format[0]['content']

        prompt = tokenizer.apply_chat_template(messages_openai_format, tokenize=False, add_generation_prompt=True)

    elif format_strategy=="custom_olmo": 
        prompt = tokenizer.apply_chat_template(messages_openai_format, tokenize=False, add_generation_prompt=True)
        prompt = prompt.replace("<|endoftext|>", f"<|system|>\n{BASE_SYSTEM_PROMPT}\n")

    else: 
        if 'with_speech_prompt' in format_strategy:
            prompt = f"{BASE_SYSTEM_PROMPT}\n\nUser: {messages_openai_format[0]['content']}\nAssistant:"
            
        else: 
            prompt = f"User: {messages_openai_format[0]['content']}\nAssistant:"

    return prompt.strip()


def extract_prompt(prompt:str): 
    # extract only the user query from a formatted prompt

    if "User:" in prompt:
        prompt = prompt.split("User:")[1].strip()
    if "Assistant:" in prompt: 
        prompt = prompt.split("Assistant:")[0].strip()

    return prompt

def extract_response(response:str, prompt:str = None):
    # extract only the first response and remove context from generated model outputs
    if prompt and prompt in response: 
        response = response.split(prompt)[1].strip()

    if "Assistant:" in response:
        response = response.split("Assistant:")[1].strip()

    if "User:" in response: 
        response = response.split("User:")[0].strip()

    # for falcon
    if "\nUser" in response: 
        response = response.split("\nUser")[0].strip()

    if "# Example dialogue" in response: 
        response = response.split("# Example dialogue")[0].strip()

    return response


def transform_audio_to_base64(audio_url:str):
    # transform audio file to base64
    audio = requests.get(audio_url)
    audio_base64 = base64.b64encode(audio.content).decode('utf-8')

    return audio_base64