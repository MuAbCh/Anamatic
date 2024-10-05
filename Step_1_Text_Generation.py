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
if __name__ == '__main__':
    concept = "moon discoveries and space"
    system_prompt = f"""You are the first step in a pipeline designed to create YouTube Shorts videos on a topic provided by the user. Your specific role is to generate a script that will later be paired with relevant visuals (images and videos). Here’s a breakdown of what you need to do:

    Receive the Topic: You will be given a topic by the user, and your job is to create a script based on that topic.

    Informative Content: Your script must be packed with clear, concise, and informative content. Each sentence should teach or explain something new about the topic, focusing on key facts, insights, or interesting aspects that add value to the viewer's understanding.

    Engaging Flow: The script should flow smoothly from one sentence to the next, keeping the audience engaged. Avoid unnecessary tangents or filler words. Keep the viewer's attention by making the information easy to follow with a conversational tone, if appropriate.

    Sentence Structure for Visuals: Ensure each sentence includes clear keywords or phrases that can easily be matched with relevant visuals. Think about what images or videos (e.g., objects, actions, or scenes) would naturally fit with each sentence. This makes it easier for the next step in the pipeline to pair the script with media.

    Keep It Short and Concise: Remember, this is for a short video. Every sentence should be efficient and impactful. Eliminate unnecessary details or repetition. Be brief, but informative.

    Avoid Redundancy: Never repeat the same information unless it's for emphasis. Each sentence should move the script forward by introducing new ideas or facts. Do not restate information already mentioned, unless it adds value in a different context or perspective. 
    """

    user_prompt = f""" 

    Write a concise script for a YouTube Short on {concept} with a max of 225 words. 
    The script must be engaging and flow smoothly, with each sentence offering new information or insights. 
    Avoid tangents or filler words, and ensure every sentence introduces something new. 
    Include visual keywords in each sentence to pair with relevant visuals (e.g., actions or scenes). 
    The script should deliver valuable information quickly without redundancy. Keep it efficient, impactful, and concise throughout.
    """

    script = generate_text(prompt_system=system_prompt, prompt_user=user_prompt)
    
    system_prompt = " Closely follow the Users instructions."
    user_prompt = f""" Here is the current script: {script}
    Remove any headings that say "Here is the script" etc. The script should start right away.
    I dont want the "visual descriptions " in the sript, the TTS model will speak the script as it is written.
    Also remove the seconds of how long the sentence is, we will calculate that on our own"""
    
    script = generate_text(prompt_system=system_prompt, prompt_user=user_prompt)

    print(script)