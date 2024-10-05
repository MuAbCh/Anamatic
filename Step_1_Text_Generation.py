# getting the API keys and setting up the inference models not stored locally:
import os
from groq import Groq
from dotenv import load_dotenv
import json

import os
import json


from utils.file_related import open_json_file, load_from_file


def generate_text(prompt_system, prompt_user, max_new_tokens= 1000, self_model="llama-3.1-70b-versatile", self_temperature=0.00):
    load_dotenv(".env")
    Groq_api_key  = os.getenv("GROQ_APIKEY")
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


system_prompt = f"""You are the first step in a pipeline designed to create YouTube Shorts videos on a topic provided by the user. Your specific role is to generate a script that will later be paired with relevant visuals (images and videos). Here’s a breakdown of what you need to do:

Receive the Topic: You will be given a topic by the user, and your job is to create a script based on that topic.

Informative Content: Your script must be packed with clear, concise, and informative content. Each sentence should teach or explain something new about the topic, focusing on key facts, insights, or interesting aspects that add value to the viewer's understanding.

Engaging Flow: The script should flow smoothly from one sentence to the next, keeping the audience engaged. Avoid unnecessary tangents or filler words. Keep the viewer's attention by making the information easy to follow with a conversational tone, if appropriate.

Sentence Structure for Visuals: Ensure each sentence includes clear keywords or phrases that can easily be matched with relevant visuals. Think about what images or videos (e.g., objects, actions, or scenes) would naturally fit with each sentence. This makes it easier for the next step in the pipeline to pair the script with media.

Keep It Short and Concise: Remember, this is for a short video. Every sentence should be efficient and impactful. Eliminate unnecessary details or repetition. Be brief, but informative.

Avoid Redundancy: Never repeat the same information unless it's for emphasis. Each sentence should move the script forward by introducing new ideas or facts. Do not restate information already mentioned, unless it adds value in a different context or perspective. 
"""

user_prompt = f""" I need you to write a script for a YouTube Short based on the topic: {concept}. Follow these guidelines carefully:
Duration: The entire script must be at most 225 words long.

Informative and Concise: The script should be packed with clear, concise information that explains or teaches the topic in an engaging way. Each sentence must provide new facts, insights, or interesting points about the topic.

Engaging Flow: Make sure the script flows smoothly from one idea to the next. Avoid tangents or irrelevant information. Keep the audience’s attention by making the content easy to follow and engaging.

Visual Keywords: Every sentence should include keywords or phrases that can easily be paired with relevant visuals (e.g., objects, actions, scenes). Think about what visuals naturally match each sentence and mention those within the script.

Short and Efficient: Keep the script concise. It should deliver valuable information quickly and avoid unnecessary details. Each sentence should be impactful and straight to the point.

Reminder for the Duration, the script must be at most 225 words long 

No Redundancy: Avoid repeating the same information. Every sentence should introduce something new and push the script forward. Only restate ideas if it adds meaningful emphasis.
"""

script = generate_text(prompt_system=system_prompt, prompt_user=user_prompt)
print(script)

