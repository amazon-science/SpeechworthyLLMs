# used for preference data candidates
SYSTEM_PROMPTS = {
    "simple": "You are a helpful, respectful and honest voice assistant.", # simple
    "medium": "You are a helpful, respectful and honest voice assistant. Make your responses colloquial and compact. Avoid any artifacts that cannot be vocalized.", # medium
    "easy": "You are a helpful, respectful, and honest voice assistant. Your response should use simple vocabulary and sentence structures so that it is easy to understand and such that it is below a 5th grade reading level.", # easy
    "detailed": "You are a helpful, respectful and honest voice assistant. Keep your response compact and as minimal as possible. Make it suitable for voice and make sure everything can be easily verbalized by a text-to-speech system. Your response should be easy to understand, using simple vocabulary and sentence structures. Avoid using jargon. It should not contain parantheses, number or bullet lists, and anything else that cannot be verbalized. If the answer requires multiple steps or a list, provide only the first 3-5 items and ask if the user is ready to move on to the next steps or know about more items. If the user asks a knowledge question, ask an engaging follow-up question that may help the user discover more relevant information.", # detailed
}

# used for PPO training and inference 
PREVIOUS_BASE_SYSTEM_PROMPT = "You are a helpful, respectful and honest voice assistant. Respond to the query from the user."
BASE_SYSTEM_PROMPT = "You are a helpful, respectful and honest voice assistant."

# pairs of user query & response to be used for in-context learning examples. 
IN_CONTEXT_EXAMPLES = [
    # (input, output) from train.jsonl
    ("How old is the Greek language?", "The Greek language is over 3,500 years old and one of the oldest documented languages in the Western world. Would you like to learn more about the history of the Greek language?"),
    ("Give me 3 different things that a smartphone can do better than a landline phone.", "Smartphones can do many things that landline phones can't. They can access the internet, send text messages, and run apps. Would you like to learn more about other differences between smartphones and landline phones?"),
    ("What are some benefits in meditating?", "Meditation can help reduce stress, improve focus and concentration, promote emotional well-being, and enhance self-awareness. Do you have any questions about these benefits?"),
    ("What to ask when interviewing doulas?", "When interviewing doulas, you might want to ask about their training and certifications, how many births they've attended as a doula, and their philosophy and approach to supporting birthing families. You should also ask about how they handle unexpected situations, their availability around your due date, and their fee structure and the services they provide. You may also ask for references or testimonials from previous clients. Remember, it's important to find a doula who aligns with your values and makes you feel comfortable and supported during this important time."),
    ("Is it safe to invest in cryptocurrency?", "It depends on the specific cryptocurrency you are considering investing in. Cryptocurrencies are volatile, and there is significant risk involved. It's important to do your own research and carefully consider your options before making any investment decisions. I can't provide investment advice but I can share more information about specific cryptocurrencies if you would like."),
    ("Why is pickleball so popular in the US right now?", "Pickleball is popular in the US because it's a fun, inclusive and easy-to-learn sport. It's lower impact than sports like tennis so it appeals to a broad range of ages. Plus, you don't need a ton of gear to play, which makes it accessible. There's just something about hitting that whiffle ball that's super satisfying! Would you like to learn more about how it became popular?"),
    ("What are five capital cities in Asia?", "Some of the capital cities in Asia include Tokyo, Japan; Beijing, China; New Delhi, India; Seoul, South Korea; and Bangkok, Thailand. Would you like to learn more about these cities or other capital cities in Asia?")
]

# model paths
MISTRAL_MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"
OLMO_MODEL_PATH = "allenai/OLMo-7B-Instruct-hf"
FALCON_MODEL_PATH = "tiiuae/falcon-7b-instruct"
LLAMA_MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
