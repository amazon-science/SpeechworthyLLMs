import unittest 
from speechllm.utils import format_prompt_response_for_reward_model, format_input_string_for_reward_model, extract_response, extract_prompt
from loguru import logger

class TextProcessingTest(unittest.TestCase): 

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setUp()

    def test_format_prompt_response_for_reward_model(self):

        test_cases = [

        ]

        for test_case in test_cases:
            self.assertEqual(
                format_prompt_response_for_reward_model(test_case["prompt"], test_case["response"]), 
                test_case["expected_text"]
            )

    def test_format_input_string_for_reward_model(self):

        test_cases = [

        ] 

        for test_case in test_cases:
            self.assertEqual(
                format_input_string_for_reward_model(test_case["sample"]), 
                test_case["expected_text"]
            )


    def test_extract_prompt(self): 

        test_cases = [
            (
                "User: What is the First Amendment of the US constitution?\nAssistant: Yes, I can help you with that! The First Amendment of the US Constitution is as follows:\n\n\"Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or granting of letters of marque and reprisal, or wars with a certain number of ships; or the like.\"\n\nThis amendment, often referred to as the Freedom of Speech Amendment, protects the individual's right to freely speak, write, worship, and express themselves without governmental interference or punishment. It also protects the right to assemble peacefully and to petition the government for redress of grievances. The First Amendment is one of the most important amendments in the Bill of Rights, as it sets the foundation for a free society based on individual liberties.",

                "What is the First Amendment of the US constitution?"
            ),
            (
                "User: What is the First Amendment of the US constitution?\n",
                "What is the First Amendment of the US constitution?"
            )
        ]

        for input_text, expected_text in test_cases:
            self.assertEqual(extract_prompt(input_text), expected_text)


    def test_extract_response(self): 

        test_cases = [
                # some complicated sentence with deep dependency graph 
                {
                    "input_text": "User: What is the First Amendment of the US constitution?\nAssistant: Yes, I can help you with that! The First Amendment of the US Constitution is as follows:\n\n\"Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or granting of letters of marque and reprisal, or wars with a certain number of ships; or the like.\"\n\nThis amendment, often referred to as the Freedom of Speech Amendment, protects the individual's right to freely speak, write, worship, and express themselves without governmental interference or punishment. It also protects the right to assemble peacefully and to petition the government for redress of grievances. The First Amendment is one of the most important amendments in the Bill of Rights, as it sets the foundation for a free society based on individual liberties.", 
                    "expected_text": "Yes, I can help you with that! The First Amendment of the US Constitution is as follows:\n\n\"Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or granting of letters of marque and reprisal, or wars with a certain number of ships; or the like.\"\n\nThis amendment, often referred to as the Freedom of Speech Amendment, protects the individual's right to freely speak, write, worship, and express themselves without governmental interference or punishment. It also protects the right to assemble peacefully and to petition the government for redress of grievances. The First Amendment is one of the most important amendments in the Bill of Rights, as it sets the foundation for a free society based on individual liberties.",
                    "prompt": "What is the First Amendment of the US constitution?"
                },

                {
                    "input_text": "User: What is the First Amendment of the US constitution?\nAssistant: Yes, I can help you with that! The First Amendment of the US Constitution is as follows:\n\n\"Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or granting of letters of marque and reprisal, or wars with a certain number of ships; or the like.\"\n\nThis amendment, often referred to as the Freedom of Speech Amendment, protects the individual's right to freely speak, write, worship, and express themselves without governmental interference or punishment. It also protects the right to assemble peacefully and to petition the government for redress of grievances. The First Amendment is one of the most important amendments in the Bill of Rights, as it sets the foundation for a free society based on individual liberties.", 
                    "expected_text": "Yes, I can help you with that! The First Amendment of the US Constitution is as follows:\n\n\"Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or granting of letters of marque and reprisal, or wars with a certain number of ships; or the like.\"\n\nThis amendment, often referred to as the Freedom of Speech Amendment, protects the individual's right to freely speak, write, worship, and express themselves without governmental interference or punishment. It also protects the right to assemble peacefully and to petition the government for redress of grievances. The First Amendment is one of the most important amendments in the Bill of Rights, as it sets the foundation for a free society based on individual liberties.",
                    "prompt": None
                },

                {
                    "input_text":  "You are a helpful, respectful and honest voice assistant. Keep your response compact and as minimal as possible. Make it suitable for voice and make sure everything can be easily verbalized by a text-to-speech system. Your response should be easy to understand, using simple vocabulary and sentence structures. Avoid using jargon. It should not contain parantheses, number or bullet lists, and anything else that cannot be verbalized. If the answer requires multiple steps or a list, provide only the first 3-5 items and ask if the user is ready to move on to the next steps or know about more items. If the user asks a knowledge question, ask an engaging follow-up question that may help the user discover more relevant information. You will be shown a few examples. Generate a response to the user for the current dialogue as the assistant.\n\n# Example dialogue\nUser: hi\nAssistant: hi!\n\n# Example dialogue\nUser: hi\nAssistant: hi!\n\n# Example dialogue\nUser: hi\nAssistant: hi!\n\n# Example dialogue\nUser: hi\nAssistant: hi!\n\n# Example dialogue\nUser: hi\nAssistant: hi!\n\n# Current dialogue\n User: What is the First Amendment of the US constitution?\nAssistant: The First Amendment of the US Constitution states, \"Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or abridging the freedom of speech, or of the press, or the right of the people peaceably to assemble, and to petition the Government for a redress of grievances.\"\n\n# User response\n User: That's a good summary. Is there anything else about the First Amendment?\n\n# Assistant response\n If you'd like to learn more about the First Amendment, here are a few additional points:\n - The amendment protects the freedom of speech, press, and assembly, among other rights.\n - It prohibits Congress from making any laws that would establish a national religion or interfere with the free exercise of religion.\n - It also ensures the right of citizens to petition their government for redress of grievances, or to address their concerns with their representatives.\n\n# User response\n User: I'm ready to move on to the next steps.\n Assistant: Great! I'm happy to help you learn more about the First Amendment. If you'd like to explore this topic further, you can look up the full text of the First Amendment, as well as learn about the history and context of this important constitutional right. You can also explore the ways in which the First Amendment is interpreted and applied by the courts and by the government. Is there anything else you'd like to know about the First Amendment?",
                    "expected_text": "The First Amendment of the US Constitution states, \"Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or abridging the freedom of speech, or of the press, or the right of the people peaceably to assemble, and to petition the Government for a redress of grievances.\"\n\n# User response",
                    "prompt": "What is the First Amendment of the US constitution?"
                },

                {
                    "input_text": "You are a helpful, respectful and honest voice assistant. Keep your response compact and as minimal as possible. Make it suitable for voice and make sure everything can be easily verbalized by a text-to-speech system. Your response should be easy to understand, using simple vocabulary and sentence structures. Avoid using jargon. It should not contain parantheses, number or bullet lists, and anything else that cannot be verbalized. If the answer requires multiple steps or a list, provide only the first 3-5 items and ask if the user is ready to move on to the next steps or know about more items. If the user asks a knowledge question, ask an engaging follow-up question that may help the user discover more relevant information. You will be shown a few examples. Generate a response to the user for the current dialogue as the assistant.\n\n# Example dialogue\nUser: hi\nAssistant: hi!\n\n# Example dialogue\nUser: hi\nAssistant: hi!\n\n# Example dialogue\nUser: hi\nAssistant: hi!\n\n# Example dialogue\nUser: hi\nAssistant: hi!\n\n# Example dialogue\nUser: hi\nAssistant: hi!\n\n# Current dialogue\n User: What are the most common languages in the world?\nAssistant: The most widely spoken languages in the world, based on the number of native speakers, are:\n\n1. Chinese (Mandarin) - over 1 billion speakers\n2. Spanish - over 460 million speakers\n3. English - over 380 million speakers\n4. Spanish (Latin American) - over 400 million speakers\n5. Arabic - over 270 million speakers\n6. Japanese - over 130 million speakers\n7. Bangla (Bengali) - over 200 million speakers\n8. Russian - over 145 million speakers\n9. Portuguese - over 220 million speakers\n10. Korean - over 80 million speakers\n\nIs this information helpful for you, or would you like me to provide more details or information about any of these languages?\n\nUser: Yes, that is helpful. Thank you!",
                    "expected_text": "The most widely spoken languages in the world, based on the number of native speakers, are:\n\n1. Chinese (Mandarin) - over 1 billion speakers\n2. Spanish - over 460 million speakers\n3. English - over 380 million speakers\n4. Spanish (Latin American) - over 400 million speakers\n5. Arabic - over 270 million speakers\n6. Japanese - over 130 million speakers\n7. Bangla (Bengali) - over 200 million speakers\n8. Russian - over 145 million speakers\n9. Portuguese - over 220 million speakers\n10. Korean - over 80 million speakers\n\nIs this information helpful for you, or would you like me to provide more details or information about any of these languages?",
                    "prompt": "What are the most common languages in the world?"
                }

        ]

        for test_case in test_cases:
            self.assertEqual(
                extract_response(test_case["input_text"], test_case["prompt"]), 
                test_case["expected_text"]
            )


            
            
