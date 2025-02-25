import unittest 
from speechllm import SpeechLLMEval
from loguru import logger

class SpeechllmEvalTest(unittest.TestCase): 

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setUp()
        self.speechllmeval = SpeechLLMEval()

    def test_get_readability_score(self): 

        test_cases = [
            # some complicated sentence that is difficult to read
            "Indefagitabundus, tanquam flavum fraticinida."

            # some easy sentence that is easy to read 
            "The quick brown fox jumped over the lazy dog."
        ]

        for test_case in test_cases:
            self.assertEqual(isinstance(self.speechllmeval.get_readability_score(test_case),float), True)


        comparative_test_cases = [
            # pairs of simple and complicated sentence that is difficult to read
            ("Indefagitabundus, tanquam flavum fraticinida.", "The quick brown fox jumped over the lazy dog.")
        ]

        for test_case in comparative_test_cases:
            readabilty_complicated = self.speechllmeval.get_readability_score(test_case[0])
            readability_simple = self.speechllmeval.get_readability_score(test_case[1])

            logger.info(f"\nreadabilty_complicated: {readabilty_complicated}\nreadability_simple: {readability_simple}")
            self.assertEqual(
                readabilty_complicated < readability_simple,
                True 
            ) 


    def test_get_dependency_graph_depth(self): 

        test_cases = [
            # some complicated sentence with deep dependency graph 
            "The cat on the mat chased the mouse in the hole under the floor of the house.",
            "I like pizza." 
        ]

        for test_case in test_cases:
            mean_depth, depths = self.speechllmeval.get_dependency_graph_depth(test_case)
            for depth in depths: 
                self.assertEqual(isinstance(depth, int), True)
            self.assertEqual(isinstance(mean_depth, float), True)
            
