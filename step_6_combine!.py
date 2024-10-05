import os
import tempfile
import shutil
import gc
import json
import datetime as dt

from utils.file_related import open_json_file, load_from_file

from google.cloud import texttospeech

import whisperx
import ffmpeg

import subprocess
from whisperx.utils import get_writer
import re

import time
from groq import Groq
from dotenv import load_dotenv

import random

import requests

import openvino_genai as ov_genai
from transformers import pipeline
import scipy
import torch
import soundfile as sf
from tqdm.auto import tqdm
import dotenv
import openai  # Ensure you have the 'openai' library installed
from pathlib import Path
from optimum.intel.openvino import OVStableDiffusionXLPipeline
from PIL import Image
import matplotlib.pyplot as plt
from openai import OpenAI
import numpy as np


from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeAudioClip,
    TextClip,
    CompositeVideoClip,
    afx  # Import audio effects for looping
)
from moviepy.audio.AudioClip import AudioClip
from moviepy.audio.AudioClip import concatenate_audioclips

import logging

import os
import random
from pydub import AudioSegment
from pydub.effects import normalize


from Step_1_Text_Generation import generate_text
from Step_2_Music_Generation import generate_music_file
from Step_3_Audio_Generation import generate_audio
from Step_4_Transcript_Generation import transcribe_audio, make_file_path
from step_5_img_gen import get_b_rolls
from step_5_video_gen import gen_video

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Step 5 images:

# List of categories and keywords (for easy reference)
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

# 1: Create a list of images with their associated keywords
image_data = []
video_data = []

# Example structure for the images; actual paths can be dynamic.
def generate_image_metadata():
    image_index = 0
    for category, keywords in categories.items():
        for i in range(3):  # 3 images per category
            # Randomly assign 2 keywords to each image
            img_keywords = random.sample(keywords, 2)
            image_data.append({
                "image_path": f"{category.replace(' ', '_')}_image_{i + 1}.png",
                "keywords": img_keywords
            })

def generate_video_metadata():
    video_index = 0
    for category, keywords in categories.items():
        # Randomly assign 2 keywords to each video
        vid_keywords = random.sample(keywords, 2)
        video_data.append({
            "video_path": f"{category.replace(' ', '_')}_video_{1}.mp4",
            "keywords": vid_keywords
        })


# Generate image metadata
generate_image_metadata()
generate_video_metadata()

# 2: Function to find the best match for a set of keywords
def find_best_video_match(query_keywords):
    """
    Given a list of three keywords, find the video that matches the most keywords.
    
    Parameters:
    - query_keywords (list): A list of 3 keywords.
    
    Returns:
    - dict: The video data (path, keywords) of the best matching video.
    """
    best_match = None
    highest_score = 0
    
    for image in video_data:
        # Calculate how many keywords match
        match_count = len(set(query_keywords).intersection(image['keywords']))
        
        if match_count > highest_score:
            best_match = image
            highest_score = match_count
    
    if best_match:
        return best_match
    else:
        return None

def find_best_match(query_keywords):
    """
    Given a list of three keywords, find the image that matches the most keywords.
    
    Parameters:
    - query_keywords (list): A list of 3 keywords.
    
    Returns:
    - dict: The image data (path, keywords) of the best matching image.
    """
    best_match = None
    highest_score = 0
    
    for image in image_data:
        # Calculate how many keywords match
        match_count = len(set(query_keywords).intersection(image['keywords']))
        
        if match_count > highest_score:
            best_match = image
            highest_score = match_count
    
    if best_match:
        return best_match
    else:
        return None

# Function to get a default image if no best match is found
def get_default_image():
    return random.choice(image_data)  # Return a random image from the dataset
def get_default_video():
    return random.choice(video_data)  # Return a random video from the dataset

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

    

import re
import logging

def update_highlight_times(srt_file, lines_to_highlight):
    """
    Sets the beginning times for the highlights in the SRT file based on the provided array of lines.

    Parameters:
    srt_file (str): The path to the SRT file.
    lines_to_highlight (list): A list of strings (lines) to highlight.

    Returns:
    list: A list of tuples where each tuple contains (highlighted line, start time in seconds).
    """
    
    # Read the SRT file and parse its contents
    with open(srt_file, 'r', encoding='utf-8') as file:
        srt_lines = file.readlines()

    # Regex pattern to extract time and text from SRT file
    time_pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})')
    srt_entries = []

    current_time = None
    current_text = ""

    for line in srt_lines:
        line = line.strip()  # Strip leading and trailing whitespace

        if time_pattern.match(line):  # This line contains the time range
            # When a new time block starts, save the previous block if it exists
            if current_time and current_text:
                # Store as a tuple of (start time, text)
                srt_entries.append((current_time, current_text.strip()))
                current_text = ""  # Reset text for the next block

            # Capture the start time
            current_time = time_pattern.match(line).group(1)
        elif line and not line.isdigit():  # Only add text lines, not empty or index lines
            current_text += " " + line
        elif not line and current_time and current_text:
            # When we hit a blank line after text, we save the current block
            srt_entries.append((current_time, current_text.strip()))
            current_text = ""
            current_time = None

    # Check for the last entry
    if current_time and current_text:
        srt_entries.append((current_time, current_text.strip()))

    # Helper function to convert SRT time format to seconds
    def srt_time_to_seconds(srt_time):
        hours, minutes, seconds_millis = srt_time.split(':')
        seconds, millis = seconds_millis.split(',')
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000

    # Sliding window match function with special handling for <u> tags
    def sliding_window_match(srt_text, key_words):
        """
        Perform a sliding window match for consecutive words in the srt_text.
        Tries to match pairs of words and progressively widen the match window if no match is found.
        """
        srt_words = srt_text.split()
        key_length = len(key_words)

        # Start with pairs of words and widen the window progressively
        for window_size in range(2, key_length + 1):  # Start with 2-word windows
            for i in range(key_length - window_size + 1):
                key_window = " ".join(key_words[i:i + window_size])  # Get sliding window of key words

                for j in range(len(srt_words) - window_size + 1):
                    srt_window = " ".join(srt_words[j:j + window_size])  # Get sliding window of srt words

                    # Check for regular match or matches with <u> tags
                    if (
                        key_window in srt_window or  # Regular match
                        any(
                            f"<u>{word}</u>" in srt_window for word in key_words[i:i + window_size]
                        )
                    ):
                        return True
        return False

    # Initialize a result list to store matched lines and their start times
    result = []

    # Iterate over each line in the array
    for line in lines_to_highlight:
        words = line.split()  # Split the line into words
        found = False  # Flag to indicate if a match was found

        # Try different word windows in SRT entries
        for entry in srt_entries:
            srt_time, srt_text = entry  # Unpack the tuple

            # Perform a sliding window match with the SRT text
            if sliding_window_match(srt_text, words):
                highlight_time = srt_time_to_seconds(srt_time)
                result.append((line, highlight_time))  # Store the result as (line, start time)
                found = True  # Set flag to indicate a match was found
                break

        if not found:
            logging.warning(f"Could not find a match for: {line}")
            result.append((line, None))  # Append None for lines that weren't matched

    return result



def assemble_video(image_paths, voice_over_path, music_path, output_path):
    """
    Assembles the final video by stitching together images with fade transitions,
    adding audio (voice-over with added silence and looping background music),
    and overlaying subtitles.

    Parameters:
    - image_paths (list): List of paths to the generated images.
    - voice_over_path (str): Path to the voice-over audio file.
    - music_path (str): Path to the background music audio file.
    - subtitles_path (str): Path to the subtitles JSON file.
    - output_path (str): Path to save the final video.

    Returns:
    - None
    """
    # Define the duration for crossfades between clips (in seconds)
    crossfade_duration = 1  # Adjust as needed

    # Load Voice-Over Audio first to determine its duration
    try:
        voice_over = AudioFileClip(voice_over_path)
    except Exception as e:
        logging.error(f"Failed to load voice-over audio: {e}")
        voice_over = None

    # Add 1-second silence to the end of the voice-over
    if voice_over:
        try:
            # Create a 1-second silent audio clip
            silence = AudioFileClip(lambda t: 0, duration=1).set_fps(voice_over.fps)
            # Concatenate the silence to the voice-over
            voice_over_extended = concatenate_audioclips([voice_over, silence])
        except Exception as e:
            logging.error(f"Failed to add silence to voice-over: {e}")
            voice_over_extended = voice_over  # Fallback to original voice_over
    else:
        voice_over_extended = None

    # Determine the total duration based on voice-over
    if voice_over_extended:
        total_duration = voice_over_extended.duration
    else:
        total_duration = 0

    # Create Image Clips with crossfade transitions
    clips = []
    for idx, image_path in enumerate(image_paths):
        try:
            clip = ImageClip(image_path).set_duration(total_duration / len(image_paths))
            # Apply crossfadein to all clips except the first
            if idx != 0:
                clip = clip.crossfadein(crossfade_duration)
            clips.append(clip)
        except Exception as e:
            logging.error(f"Failed to create ImageClip for {image_path}: {e}")

    if not clips:
        logging.error("No valid image clips were created.")
        return

    # Concatenate image clips with crossfade transitions
    try:
        video = concatenate_videoclips(
            clips,
            method="compose",
            padding=-crossfade_duration  # Overlap clips by crossfade_duration
        )
    except Exception as e:
        logging.error(f"Failed to concatenate video clips: {e}")
        return

    # Set the final video duration
    video = video.set_duration(total_duration)

    # Step 2: Music Editing and Looping (from he method)
    def increase_volume_at_start(audio, duration=500):  # 500 ms = 0.5 seconds
        initial = audio[:duration]
        rest = audio[duration:]
        initial = initial + 6  # Increase volume by 6 dB
        return initial + rest

    # Add Background Music and loop it to match the video duration
    try:
        # Load the music file using pydub
        audio = AudioSegment.from_mp3(music_path)

        # Determine if the file name contains "edit"
        if "edit" in os.path.basename(music_path):
            # Keep only the last x seconds (where x = total_duration)
            start_time = len(audio) - (total_duration * 1000)  # convert seconds to milliseconds
            edited_audio = audio[start_time:]
            # Increase the volume at the start
            edited_audio = increase_volume_at_start(edited_audio, 1500)
        else:
            # For non-edit files, keep only the first seconds (based on total video duration)
            edited_audio = audio[:total_duration * 1000]

        # Export the edited audio to a temporary file
        temp_music_path = "temp_edited_music.mp3"
        edited_audio.export(temp_music_path, format="mp3")

        # Load the edited music with moviepy
        music = AudioFileClip(temp_music_path).volumex(0.1)  # Lower volume for background music

        # Loop the music to ensure it covers the entire video duration
        music = music.fx(afx.audio_loop, duration=total_duration)

    except Exception as e:
        logging.error(f"Failed to load or loop background music: {e}")
        music = None

    # Combine Voice-Over and Background Music
    if voice_over_extended and music:
        final_audio = CompositeAudioClip([voice_over_extended, music])
    elif voice_over_extended:
        final_audio = voice_over_extended
    elif music:
        final_audio = music
    else:
        final_audio = None

    if final_audio:
        video = video.set_audio(final_audio)
    else:
        logging.warning("No audio was set for the video.")

    # Write the final video to the specified output path
    try:
        video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            threads=4,
            preset='medium',
            bitrate="5000k",
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
    except Exception as e:
        logging.error(f"Failed to write the final video file: {e}")
    finally:
        # Cleanup resources
        del video
        if voice_over:
            del voice_over
        if voice_over_extended:
            del voice_over_extended
        if music:
            del music
        if final_audio:
            del final_audio
        gc.collect()
# Load environment variables from .env file
dotenv.load_dotenv()

concept = input("Enter the topic you would like to make videos about: ")  # GET USER INPUT
# Sanitize the concept name to avoid issues with spaces
concept = concept.replace(" ", "_").lower()

# Create the full directory path
concept_dir = os.path.join(".", concept)

# Create the directory if it doesn't exist
os.makedirs(concept_dir, exist_ok=True)


# TODO: Write the prompt, that genrates a more detailed prompt for making the script.
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


system_prompt = " Closely follow the Users instructions."
user_prompt = f""" Here is the current script: {script}
Remove any headings that say "Here is the script" etc. The script should start right away.
I dont want the "visual descriptions " in the sript, the TTS model will speak the script as it is written.
Also remove the seconds of how long the sentence is, we will calculate that on our own"""

script = generate_text(prompt_system=system_prompt, prompt_user=user_prompt)


voice_over_path = generate_audio(text = script, person=1, dir = concept)


#########################################################################################
# Process the script and get the assigned keywords
sentence_keywords_cleaned = process_script_cleaned(script, keyword_list)

selected_images = {}

for index, (sentence, keywords) in enumerate(sentence_keywords_cleaned.items()):
    if index == 0 or index == len(sentence_keywords_cleaned) - 1:
        # For the first and last sentences, video is selected insteade of image
        best_video = find_best_video_match(keywords)
        if not best_video:
            best_video = get_default_video()
        selected_images[sentence] = best_video['video_path']
    else:
        best_image = find_best_match(keywords)
        
        # If no best match found, get a default image
        if not best_image:
            best_image = get_default_image()
        
        selected_images[sentence] = best_image['image_path']

 
#############################################################################


# Step 2: Generate Music Description
# have a textual description of the music we want. 
# TODO: just have a couple good tracks and edit them up.
# List of available music files
music_files = [
    "BRAZILIAN DANÇA PHONK.mp3",
    "Double Life (From _Despicable Me 4_).mp3",
    "edit another piano.mp3",
    "edit_audio_2.mp3",
    "edit_good_ending.mp3",
    "edit_piano.mp3"
]

# Set total_duration (replace with actual video length in seconds)
# total_duration = 60  # Example video length

# Choose a random music file
selected_file = random.choice(music_files)

music_file_path = f'music/{selected_file}'



###########################################################################
# logging.info(f"Background music saved at {music_path}")

# Step 3: Generate Voice-Over Audio
voice_over_path = generate_audio(text = script, person=1, dir = concept)
logging.info(f"Voice-over audio saved at {voice_over_path}")

# Step 4: Transcribe Audio to Generate Subtitles
# Example: voice_over_path might be something like 'new_concept/new_audio.wav'
# Join the path with the current directory for consistent path handling
voice_over_path = os.path.join(os.getcwd(), voice_over_path)

# Get just the file name (without the path) for printing purposes
audio_file = os.path.basename(voice_over_path)
print(f"Generated the Audio file: {audio_file}")

# Generate the subtitle file path
subtitle_file = make_file_path(concept, 'srt')  # Ensuring it's the full path

# Ensure you pass the full path when calling transcribe_audio
transcription_path = transcribe_audio(audio_file=voice_over_path, subtitle_file=subtitle_file, concept=concept)
logging.info(f"Transcription saved at {transcription_path}")


# Step 7: Generate Images # TODO: Each time an (video+image) set is generated, we add a desctription with the key words
model_dir = os.path.join(temp_dir, "openvino-sd-xl-base-1.0")
base_pipeline = setup_sdxl_base_model(model_dir=model_dir, device="CPU")
images_paths = generate_images(images_json_path, base_pipeline, temp_dir)
logging.info(f"Generated images: {images_paths}")

# # Step 8: Generate Subtitles File
# subtitles_path = generate_subtitles(transcription_path, temp_dir)
# logging.info(f"Subtitles saved at {subtitles_path}")


import os
from moviepy.config import change_settings

change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

# Step 9: Assemble the Video
temp_dir = "temp"
voice_over_path = os.path.join(temp_dir, "voice_over.wav")
music_path = os.path.join(temp_dir, "background_music.wav")
# subtitles_path = os.path.join(temp_dir, "subtitles.json")
images_paths = [os.path.join(temp_dir, f"image_{i}.png") for i in range(0, 5)]  # Assuming 5 images
output_video_path = os.path.join(temp_dir, "final_video.mp4")

assemble_video(images_paths, voice_over_path, music_file_path, output_video_path)
logging.info(f"Final video saved at {output_video_path}")

# Move the final video to the current directory
final_output = os.path.join(os.getcwd(), "final_video.mp4")
shutil.move(output_video_path, final_output)
logging.info(f"Video moved to {final_output}")

logging.info("All done!")