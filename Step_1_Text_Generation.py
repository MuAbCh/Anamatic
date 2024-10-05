# getting the API keys and setting up the inference models not stored locally:
import os
print("hello")
from groq import Groq
from dotenv import load_dotenv
import json
from utils.file_related import open_json_file, load_from_file
# changed this parameters, system_prompt and user_prompt
def generate_text(prompt_system, prompt_user, max_new_tokens= 1000, self_model="llama-3.1-70b-versatile", self_temperature=0.00):
    load_dotenv(".env")
    Groq_api_key  = os.getenv("GROQ_APIKEY")
    hf_token = os.getenv("HF_TOKEN")
    client = Groq(api_key = Groq_api_key,)
    chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content":prompt_system
                },
                {
                    "role": "user",
                    "content": prompt_user,
                }
            ],
            model=self_model, temperature=self_temperature,
        )
    # returns just the text
    return chat_completion.choices[0].message.content


# testing the function
system_prompt = """
You are a helpful assistant that generates song prompts for sonu.ai """

user_prompt = f"""
Can you give me a comma separated description of the song Blinding Lights by The Weeknd in 200 characters, 
including spaces for a song prompt for sonu.ai? You can't use the artist name so describe the vocals too.
"""
script = generate_text(system_prompt, user_prompt)
print(script)