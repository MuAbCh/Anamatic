import time
import requests
import os
from Step_1_Text_Generation import generate_text
# Replace your Vercel domain
base_url = 'https://suno-api-eight-eta.vercel.app/'

def custom_generate_audio(payload):
    url = f"{base_url}/api/custom_generate"
    response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
    return response.json()

def extend_audio(payload):
    url = f"{base_url}/api/extend_audio"
    response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
    return response.json()

def generate_audio_by_prompt(payload):
    url = f"{base_url}/api/generate"
    response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
    return response.json()

def get_audio_information(audio_ids):
    url = f"{base_url}/api/get?ids={audio_ids}"
    response = requests.get(url)
    return response.json()

def get_quota_information():
    url = f"{base_url}/api/get_limit"
    response = requests.get(url)
    return response.json()

def get_clip(clip_id):
    url = f"{base_url}/api/clip?id={clip_id}"
    response = requests.get(url)
    return response.json()

def generate_whole_song(clip_id):
    payloyd = {"clip_id": clip_id}
    url = f"{base_url}/api/concat"
    response = requests.post(url, json=payload)
    return response.json()

# Function to save audio file locally in a specific directory
def download_audio_file(url, file_name, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Full file path
    file_path = os.path.join(output_dir, file_name)
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file_path}")
    else:
        print(f"Failed to download {file_name}")
    return file_path

# Function to generate and download music files to a specific directory
def generate_music_file(prompt, make_instrumental=False, wait_audio=False, retries=60, retry_delay=5, output_dir="./audio_files"):
    """
    Generates music based on a given text prompt and downloads the resulting audio files.

    Parameters:
    -----------
    prompt : str
        The text prompt used to generate the audio. This should describe the style, mood, and content of the music.
        
    make_instrumental : bool, optional (default=False)
        If True, the generated audio will be instrumental without vocals. If False, it may include vocals.

    wait_audio : bool, optional (default=False)
        If True, the function will wait for the audio generation to complete before returning a response. If False, it will return as soon as the request is initiated.

    retries : int, optional (default=60)
        The maximum number of times to poll the server to check if the audio files are ready for download. Each retry is followed by a delay specified by `retry_delay`.

    retry_delay : int, optional (default=5)
        The number of seconds to wait between each retry while polling for the audio files' status.

    output_dir : str, optional (default="./audio_files")
        The directory where the audio files will be downloaded and stored. If the directory does not exist, it will be created.

    Returns:
    --------
    tuple of str
        Returns a tuple containing the paths of the downloaded audio files (audio_1.mp3, audio_2.mp3).
        If audio generation fails or times out, it returns None.
    """
    
    # Generate audio based on the provided prompt
    data = generate_audio_by_prompt({
        "prompt": prompt,
        "make_instrumental": make_instrumental,
        "wait_audio": wait_audio
    })
    import ipdb; ipdb.set_trace()
    # Combine the IDs of generated audio
    ids = f"{data[0]['id']},{data[1]['id']}"
    
    # Poll to check the status of the generated audio files
    for _ in range(retries):
        audio_data = get_audio_information(ids)
        if audio_data[0]["status"] == 'streaming':
            audio_url_1 = audio_data[0]['audio_url']
            audio_url_2 = audio_data[1]['audio_url']
            
            # Download the audio files to the specified directory
            audio_file_1 = download_audio_file(audio_url_1, 'audio_1.mp3', output_dir)
            audio_file_2 = download_audio_file(audio_url_2, 'audio_2.mp3', output_dir)
            return audio_file_1, audio_file_2
        
        # Wait for the specified delay before checking again
        time.sleep(retry_delay)
    
    # Return None if the audio files could not be retrieved
    return None

# Example usage: call the generate_music_file function and specify the output directory
if __name__ == '__main__':

    system_prompt = ''' You are going to be outputting a prompt that will be used by Suno AI, an music generating LLM, 
    to create a tone/beat.
    This beat will be overlayed on top of a video and will be explaining a concept/ interesting idea to an audience.
    The music should be 60 seconds long, and be an instrumental type music with very few lyrics.
    Basically you give me a comma separated description of the song {The song will be given in user prompt} in 200 characters, including spaces for a song prompt for sonu.ai. You can't use the artist name so describe the vocals too. 
    The song should start strong for the introductoin and then mellow down for the body of the video , and then end strong again for the final ending.
    The description must be:
    200 characters including spaces, otherwise it will fail
    can't use the artist name
    Describe the vocals too.
    HAS TO BE A comma separated description
    Again, HAS TO BE A comma separated description

'''
    user_prompt = '''Can you give me a description that is 200 characters, including spaces for a song prompt for sonu.ai. You can't use 
                    the artist name so describe the vocals too. Make the song upbeat at the start, mellow in the middle, and then upbeat again at the end. 
                    The song must be 60 seconds long'''
    




    prompt_text = generate_text(prompt_system=system_prompt, prompt_user=user_prompt)
    
    output_directory = "music"  # Specify your desired directory here
    
    audio_files = generate_music_file(prompt_text, make_instrumental=True, wait_audio=False, output_dir=output_directory)

    if audio_files:
        print(f"Audio files downloaded: {audio_files[0]}")
    else:
        print("Failed to generate and download audio.")
