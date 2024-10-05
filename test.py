from Step_1_Text_Generation import generate_text
import re

# Assuming you already have the function `generate_text()` from your previous code
# Sample list of keywords (You can modify it with the actual list of 44 keywords)
keyword_list = [
    "Serenity", "Vibrant", "Expanse", "Majestic", "Skyline", "Symmetry", "Modern", "Monumental",
    "Cosmic", "Orbital", "Astronomical", "Expedition", "Predator", "Herd", "Exotic", "Natural Habitat",
    "Geometric", "Colorful", "Surreal", "Chaotic", "Revolutionary", "Ancient", "Conflict", "Iconic",
    "Mystical", "Legendary", "Castle", "Sorcery", "Futuristic", "Interactive", "Automated", "Wearable",
    "Vibrant Costumes", "Fireworks", "Parades", "Rituals", "Dynamic", "Victory", "Endurance", "Teamwork",
    "Algorithm", "Debugging", "Syntax", "Automation"
]

# Step 1: Function to split the script into sentences
def split_script_into_sentences(script):
    sentences = re.split(r'(?<=[.!?]) +', script)
    return sentences

# Step 2: Generate keywords for each sentence by calling the `generate_text()` function
def assign_keywords_to_sentence(sentence, keyword_list, model="llama-3.1-70b-versatile", temperature=0.00):
    system_prompt = "You are tasked with assigning two relevant keywords from a predefined list to the given sentence."
    user_prompt = f"""Here is the list of available keywords: {', '.join(keyword_list)}. 
    Assign two keywords from this list that best describe the following sentence: "{sentence}"."""
    
    keywords = generate_text(system_prompt, user_prompt, self_model=model, self_temperature=temperature)
    
    # Assuming the model returns the keywords separated by commas or in some parsable format
    assigned_keywords = keywords.split(", ")
    return assigned_keywords

# Step 3: Process the entire script
# Function to clean up the output from generate_text and extract only keywords
def clean_keywords(assigned_keywords):
    # Join the list of assigned keywords into a single string
    joined_keywords = ' '.join(assigned_keywords)
    
    # Use regex to extract only words that match the known keyword list
    # Here, we're assuming that keywords are proper nouns (capitalize them)
    cleaned_keywords = re.findall(r'\b[A-Z][a-zA-Z]*\b', joined_keywords)

    return cleaned_keywords

# Process the entire script with cleaned keywords
def process_script_cleaned(script, keyword_list):
    sentences = split_script_into_sentences(script)
    sentence_keywords = {}

    for i, sentence in enumerate(sentences):
        # Get raw keywords from generate_text
        assigned_keywords_raw = assign_keywords_to_sentence(sentence, keyword_list)
        
        # Clean the keywords by removing the extra words
        cleaned_keywords = clean_keywords(assigned_keywords_raw)
        
        sentence_keywords[sentence] = cleaned_keywords

    return sentence_keywords

# Step 4: Example usage of the function
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
    # Process the script and get the assigned keywords
    sentence_keywords_cleaned = process_script_cleaned(script, keyword_list)
    
    # Print the sentences and their assigned keywords
    for sentence, keywords in sentence_keywords_cleaned.items():
        print(f"Sentence: {sentence}")
        print(f"Assigned Keywords: {keywords}")
        print()
