from speechllm.models import FalconGenerativeModel
from speechllm.utils import form_openai_input_messages
from loguru import logger
import unittest
import time

class FalconGenerativeModelTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.falcon = FalconGenerativeModel(from_hf_directly=True, model_size="7b")

    def test_format_response(self): 

        system_prompt = "You are a helpful voice assistant."
        user_prompt = "what is the recipe for mayonnaise?"
        conversation_history = form_openai_input_messages(system_prompt, user_prompt)
        formatted_input = self.falcon.form_input_prompt(conversation_history)

        assert formatted_input == f"{system_prompt}\n{self.falcon.USER_NAME}: {user_prompt}\n{self.falcon.BOT_NAME}:"

if __name__ == "__main__":
        
    falcon = FalconGenerativeModel(from_hf_directly=True, model_size="40b")

    while True: 

        system_prompt = input("Enter a system prompt: ")

        if system_prompt == "quit":
            break
        if system_prompt =="": 
            system_prompt = "You are a helpful voice assistant. Only generate a response as the assistant."

        user_prompt = input("Enter a user prompt: ")
        if user_prompt == "": 
            user_prompt = "what is the recipe for mayonnaise?"

        logger.info(f"System prompt: {system_prompt}, User prompt: {user_prompt}")
        conversation_history = form_openai_input_messages(system_prompt, user_prompt)

        start = time.time()
        formatted_input = falcon.form_input_prompt(conversation_history)
        sequences = falcon.generate_response(conversation_history, temperature=0.4, max_length=2048, num_return_sequences=1)
        end = time.time()
        # get elapsed time in seconds
        elapsed = end - start
        logger.info(f"Time elapsed: {elapsed}s")

        for seq in sequences:
            breakpoint() 
            index = seq["generated_text"].find(falcon.BOT_NAME)
            print(f"Result: {seq['generated_text']}")
            


        