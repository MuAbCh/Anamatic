import requests
import json
from dotenv import load_dotenv
import os
import time

load_dotenv(".env")

STABLE_DIFFUSION_API_KEY = os.getenv("STABLE_DIFFUSION_API_KEY")

categories = {
    # "Nature Landscapes": ["Serenity", "Vibrant", "Expanse", "Majestic"],
    # "Urban Architecture": ["Skyline", "Symmetry", "Modern", "Monumental"],
    # "Space Exploration": ["Cosmic", "Orbital", "Astronomical", "Expedition"],
    # "Animals in the Wild": ["Predator", "Herd", "Exotic", "Natural Habitat"],
    "Abstract Art": ["Geometric", "Colorful", "Surreal", "Chaotic"],
    "Historical Events": ["Revolutionary", "Ancient", "Conflict", "Iconic"],
    "Fantasy Worlds": ["Mystical", "Legendary", "Castle", "Sorcery"],
    "Technology and Gadgets": ["Futuristic", "Interactive", "Automated", "Wearable"],
    "Cultural Festivals": ["Vibrant Costumes", "Fireworks", "Parades", "Rituals"],
    "Sports in Action": ["Dynamic", "Victory", "Endurance", "Teamwork"],
    "Programming Concepts": ["Algorithm", "Debugging", "Syntax", "Automation"]
}

def generate_video(prompt, negative_prompt, seconds):
    url = "https://stablediffusionapi.com/api/v5/text2video"

    payload = json.dumps({
        "key": STABLE_DIFFUSION_API_KEY,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "scheduler": "UniPCMultistepScheduler",
        "seconds": seconds
    })

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload)
    return json.loads(response.text)

def fetch_result(request_id):
    url = "https://stablediffusionapi.com/api/v4/dreambooth/fetch"

    payload = json.dumps({
        "key": STABLE_DIFFUSION_API_KEY,
        "request_id": request_id
    })

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload)
    return json.loads(response.text)

def gen_video(prompt1, negative_prompt1, seconds1, dir):
    prompt = prompt1
    negative_prompt = negative_prompt1
    seconds = seconds1

    # Initial request to generate video
    response_data = generate_video(prompt, negative_prompt, seconds)

    # Check if the request was accepted for processing
    if response_data['status'] == 'processing':
        print("Video generation started. Waiting for completion...")
        request_id = response_data['id']
        
        # Poll for results
        while True:
            time.sleep(10)  # Wait for 10 seconds before each check
            result = fetch_result(request_id)
            
            if result['status'] == 'success':
                video_url = result['output'][0]  # Assuming the first output is the video URL
                break
            elif result['status'] == 'failed':
                print("Video generation failed:", result.get('message', 'Unknown error'))
                return
            else:
                print("Still processing... ETA:", result.get('eta', 'Unknown'))

        # Download the video
        print("Downloading video...")
        video_response = requests.get(video_url)
        
        # Generate a filename
        filename = f"{dir}/generated_video3.mp4"
        
        # Save the video to the current working directory
        with open(filename, 'wb') as f:
            f.write(video_response.content)
        
        print(f"Video saved as {filename} in the current working directory.")
    else:
        print("Error initiating video generation:", response_data.get('message', 'Unknown error'))

    return filename

