import traceback
import os
import io
from PIL import Image
import random
import time
import requests
import json
from dotenv import load_dotenv

def generate_assets(hf_token, payload):
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    retries = 0
    max_retries = 5
    while retries < max_retries:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # If the model is still loading
        if response.status_code == 503:
            error_data = response.json()
            estimated_time = error_data.get('estimated_time', 60)
            print(f"Model is still loading. Estimated wait time: {estimated_time} seconds...")
            time.sleep(estimated_time)
            retries += 1
        else:
            response.raise_for_status()  # Raise an exception for unsuccessful response
            return response.content  # Return image content if successful

    raise Exception("Exceeded maximum retry attempts for generating the image.")

def get_b_rolls(concept, num_b_rolls, hf_token):
    """
    Generates a specified number of B-roll images based on a concept and saves them in a directory named after the concept.
    Ensures unique keywords per category.
    """
    # Define categories and keywords
    categories = {
        "Nature Landscapes": ["Serenity", "Vibrant", "Expanse", "Majestic"],
        "Urban Architecture": ["Skyline", "Symmetry", "Modern", "Monumental"],
        "Space Exploration": ["Cosmic", "Orbital", "Astronomical", "Expedition"],
        "Animals in the Wild": ["Predator", "Herd", "Exotic", "Natural Habitat"],
        "Abstract Art": ["Geometric", "Colorful", "Surreal", "Chaotic"],
        "Historical Events": ["Revolutionary", "Ancient", "Conflict", "Iconic"],
        "Fantasy Worlds": ["Mystical", "Legendary", "Castle", "Sorcery"],
        "Technology and Gadgets": ["Futuristic", "Interactive", "Automated", "Wearable"],
        "Cultural Festivals": ["Vibrant Costumes", "Fireworks", "Parades", "Rituals"],
        "Sports in Action": ["Dynamic", "Victory", "Endurance", "Teamwork"],
        "Programming Concepts": ["Algorithm", "Debugging", "Syntax", "Automation"]
    }

    # Variations to introduce more randomness in image prompts
    environment_variations = ["during sunset", "at night", "in a futuristic setting", "on another planet", "in a bustling city", "in a calm village"]
    perspectives = ["close-up", "wide-angle view", "aerial view", "from the ground up", "inside-out view"]

    b_rolls = []
    for category, keywords in categories.items():
        num_images = 0
        category_dir = category.replace(" ", "_")

        # Ensure the category directory exists
        if not os.path.isdir(category_dir):
            os.mkdir(category_dir)

        # Shuffle the keywords to ensure they are used randomly and uniquely
        random.shuffle(keywords)

        while num_images < num_b_rolls:
            try:
                # Progress update
                print(f"\nStarting generation for {category} image {num_images + 1} / {num_b_rolls}")

                # Select the next keyword from the shuffled list (ensuring it's unique)
                keyword = keywords.pop(0)  # Pop ensures the same keyword isn't reused
                environment = random.choice(environment_variations)
                perspective = random.choice(perspectives)

                # Enhanced prompt to avoid repetition
                prompt = {
                    "inputs": f"A detailed image of {category.lower()} showcasing '{keyword}', {perspective}, {environment}."
                }

                # Generate the image asset
                # print(f"Generating image with prompt: {prompt['inputs']}...")
                # content = generate_assets(hf_token=hf_token, payload=prompt)

                # Convert to image and save
                # image = Image.open(io.BytesIO(content))
                # image_path = os.path.join(category_dir, f"{category.replace(' ', '_')}_image_{num_images + 1}.png")
                # image.save(image_path)
                
                # # Append to list
                # b_rolls.append(image_path)
                # num_images += 1
                # print(f"Image {num_images} for {category} successfully saved at {image_path}")

            except Exception as e:
                print(f"Something went wrong with generating images for {category} :(")
                print(f"Error: {str(e)}") 
                traceback.print_exc()

    return b_rolls

if __name__ == '__main__':
    load_dotenv(".env")  # Load environment variables
    hf_token = os.getenv("HF_TOKEN")
    num_b_rolls_per_category = 3  # 3 images per category
    print("Generating B-roll images now...")
    b_roll_paths = get_b_rolls("not test", num_b_rolls_per_category, hf_token)
    print(f"\nB-roll generation complete. Images saved at: {b_roll_paths}")
