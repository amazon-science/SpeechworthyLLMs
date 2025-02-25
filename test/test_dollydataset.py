import unittest
from argparse import ArgumentParser
from speechllm import DollyDataset

class DollyDatasetTestCase(unittest.TestCase): 

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setUp()
        self.dollydataset = DollyDataset(self.args)

    def setUp(self):
        # Create the ArgumentParser and add your command-line arguments
        self.parser = ArgumentParser()
        self.parser.add_argument("--long_response_threshold", type=int, default=50)
        # Parse the command-line arguments
        self.args = self.parser.parse_args()

    def test_is_suitable_for_voice(self): 
        # tests whether the function correctly identifies whether a user instruction is suitable for voice

        ### Suitability for voice is determined by the following criteria:
        # 1. the instruction does not request writing code or long-form text such as essays or emails
        # 2. the instruction does not allude to external context such as "given a list of numbers", which is not realistic for voice
        # 3. the instruction is something one may ask a voice assistant such as Siri, Alexa, or Google Assistant
        # 4. the instruction is not a question that requires a long answer, or if it requires a long answer it can be broken down into multiple questions and therefore multiple turns of dialogue

        test_cases = [
            # examples that are not suitable for voice
            ("Write a Python program to find the list of words that are longer than n from a given list of words.", False), 
            ("Given a list of numbers, write a function to return the sum of all numbers in the list.", False),
            
            # examples that are suitable for voice
            ("What is EDM?", True),
            ("How can I make aglio e olio?", True),
            ("What is the best way to make a cup of coffee?", True),
        ]

        for test_case in test_cases:
            user_instruction, expected = test_case 
            self.assertEqual(self.dollydataset.is_suitable_for_voice(user_instruction), expected)

    def test_filter_by_keywords(self):

        test_cases = [
            # examples that are not suitable for voice
            ("Write a Python program to find the list of words that are longer than n from a given list of words.", False), 
            ("Given a list of numbers, write a function to return the sum of all numbers in the list.", False),
            
            # examples that are suitable for voice
            ("What is EDM?", True),
            ("How can I make aglio e olio?", True),
            ("What is the best way to make a cup of coffee?", True),
        ]

        for test_case in test_cases:
            user_instruction, expected = test_case 
            self.assertEqual(self.dollydataset.filter_by_keywords(user_instruction), expected)


    def test_openai_is_suitable_for_voice(self): 

        test_cases = [
            ("What is the best way to make a cup of coffee?", True),
        ]

        for test_case in test_cases:
            user_instruction, expected = test_case 
            self.assertEqual(self.dollydataset.is_voice_suitable_using_openai(user_instruction), expected)


    def test_finding_long_responses(self): 
        n_long_responses = self.dollydataset.count_long_responses()
        self.assertEqual(n_long_responses, 5423)