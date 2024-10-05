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
    ## Description:
    Generates a specified number of B-roll images based on a concept and saves them in a directory named after the concept.

    ## Parameters:
    - `concept` (str): The concept or theme to be included in the B-roll images.
    - `num_b_rolls` (int): The number of B-roll images to generate.
    - `hf_token` (str): The Hugging Face API token used to generate images.

    ## Returns:
    - `list`: A list of paths to the generated B-roll images.
    """
    funny_ppl = [
        "Barney the purple dinosaur from 'Barney and Friends childrens show'",
        "Donald Trump", "Snoop Dogg", "Mickey Mouse", "Goofy", 
        "Courage the Cowardly Dog", "Mr. Bean", "Teletubbies", 
        "LEGO Batman", "Kung Fu Panda", "Dumbo the elephant", 
        "Winnie the Pooh", "Buzz Lightyear", "Woody from Disney's Toy Story"
    ]

    b_rolls = []
    num_images = 0
    concept_dir = concept.replace(" ", "_")

    # Ensure the concept directory exists
    if not os.path.isdir(concept_dir):
        os.mkdir(concept_dir)

    while num_images < num_b_rolls:
        try:
            # Progress update
            print(f"\nStarting generation for B-roll image {num_images + 1} / {num_b_rolls}")
            
            # Randomly select a person from the funny_ppl list
            person = random.choice(funny_ppl)
            prompt = {"inputs": f"{person} on a computer coding with the screen saying: {concept}"}
            
            # Generate the image asset
            print(f"Generating image with prompt: {person} on a computer coding...")
            content = generate_assets(hf_token=hf_token, payload=prompt)
            
            # Convert to image and save
            image = Image.open(io.BytesIO(content))
            image_path = os.path.join(concept_dir, f"b_roll_{num_images + 1}.png")
            image.save(image_path)
            
            # Append to list
            b_rolls.append(image_path)
            num_images += 1
            print(f"Image {num_images} successfully saved at {image_path}")

        except Exception as e:
            print("Something went wrong with generating your B-roll images :(")
            print(f"Error: {str(e)}") 
            traceback.print_exc()

    return b_rolls

if __name__ == '__main__':
    load_dotenv(".env")  # Load environment variables
    concept = "not test"  # User-defined concept
    hf_token = os.getenv("HF_TOKEN")
    num_b_rolls = 3
    print("Generating B-roll images now...")
    b_roll_paths = get_b_rolls(concept, num_b_rolls, hf_token)
    print(f"\nB-roll generation complete. Images saved at: {b_roll_paths}")
