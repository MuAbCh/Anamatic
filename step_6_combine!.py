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
from Step_4_Transcript_Generation import transcribe_audio
from step_5_img_gen import get_b_rolls
from step_5_video_gen import gen_video

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Step 5 images:
def select_images():
    # based on the genre of the concpet, 
    # get a description based ont he prompt given, which  matches the best with an image's description
    # and use that image for the length of the sentence. 
    print("hello")


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

    # # Load subtitles
    # try:
    #     with open(subtitles_path, 'r') as f:
    #         subtitles = json.load(f)
    # except Exception as e:
    #     logging.error(f"Failed to load subtitles: {e}")
    #     subtitles = []

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
            silence = AudioClip(lambda t: 0, duration=1).set_fps(voice_over.fps)
            # Concatenate the silence to the voice-over
            voice_over_extended = concatenate_audioclips([voice_over, silence])
        except Exception as e:
            logging.error(f"Failed to add silence to voice-over: {e}")
            voice_over_extended = voice_over  # Fallback to original voice_over
    else:
        voice_over_extended = None

    # # Determine the total duration of the video based on subtitles and extended voice-over
    # subtitles = False
    # if subtitles:
    #     subtitles_max_end = max(sub['end'] for sub in subtitles)
    # else:
    subtitles_max_end = 0

    if voice_over_extended:
        voice_over_duration = voice_over_extended.duration
    else:
        voice_over_duration = 0

    # Calculate total_duration without buffer
    total_duration = voice_over_duration

    logging.debug(f"Voice-Over Extended Duration: {voice_over_duration} seconds")
    logging.debug(f"Total Duration: {total_duration} seconds")

    # TODO: Combine the videos and the images together (if we have a 4 sec vid, then have a 2 sec image to go along side it that way we have a 1 minuet video
    # with exactly 10 of these sets of images and videos.)
    # Calculate the duration each image will be displayed # WE HAVE TO CHANGE THIS
    # Change this also depending on the video length
    num_images = len(image_paths)
    if num_images > 0:
        # Adjust image_duration to account for crossfades 
        # TODO: Change the duration fo the video, based on the sentence length
        image_duration = (total_duration + crossfade_duration * (num_images - 1)) / num_images 
    else:
        logging.error("No images provided to assemble into video.")
        return

    logging.debug(f"Image Duration: {image_duration} seconds")

    # Create Image Clips with crossfade transitions
    clips = []
    for idx, image_path in enumerate(image_paths):
        try:
            clip = ImageClip(image_path).set_duration(image_duration)
            # Apply crossfadein to all clips except the first
            if idx != 0:
                clip = clip.crossfadein(crossfade_duration)
            clips.append(clip)
        except Exception as e:
            logging.error(f"Failed to create ImageClip for {image_path}: {e}")

    if not clips:
        logging.error("No valid image clips were created.")
        return

    # Concatenate clips with crossfade transitions
    try:
        video = concatenate_videoclips(
            clips,
            method="compose",
            padding=-crossfade_duration  # Overlap clips by crossfade_duration
        )
    except Exception as e:
        logging.error(f"Failed to concatenate video clips: {e}")
        return

    # Set the final video duration precisely
    video = video.set_duration(total_duration)

    ### FOR THE MUSIC


    def increase_volume_at_start(audio, duration=500):  # 500 ms = 0.5 seconds
        initial = audio[:duration]
        rest = audio[duration:]
        initial = initial + 6  # Increase volume by 6 dB
        return initial + rest
    # Add Background Music and loop it to match the video duration
    try:
        # Determine if the file name contains "edit"
        if "edit" in music_path.split("/")[1]:
            # Keep only the last x seconds (where x = total_duration)
            start_time = len(music_path) - (total_duration * 1000)  # convert seconds to milliseconds
            edited_audio = music_path[start_time:]
            
            # Increase the volume at the start
            edited_audio = increase_volume_at_start(edited_audio, 1500)
        else:
            # For non-edit files, keep only the first seconds (based on video length)
            edited_audio = music_path[:total_duration * 1000]

        # Save the processed audio in the variable 'music_path'
        music_path = edited_audio


        music = AudioFileClip(music_path).volumex(0.1)  # Lower volume for background music
        # Loop the music to ensure it covers the entire video duration using afx.audio_loop
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
            threads=4,  # Adjust based on your CPU
            preset='medium',
            bitrate="5000k",  # Adjust as needed
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
with tempfile.TemporaryDirectory() as temp_dir:

temp_dir = "temp"
logging.info(f"Using temporary directory: {temp_dir}")

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


# TODO: Getting the key words of the images and then ask gpt to give us what key words 
# A sentence match best with a image keyword.
# TODO: make a dictoinary of each image path, and its key words, that way we can match which key words 
# Step 5: Generate Image Descriptions
images_json_path = generate_image_descriptions(transcription_path, temp_dir)
logging.info(f"Image descriptions saved at {images_json_path}")
#  Match best wit hthe image, and get the image path stored in a list
# TODO: This list we can then put in to assemble function   
#############################################################################
# Step 1: Generate Text Script
prompt = """\n\n

"""
user_input = prompt.format(input_sentence="Spaceships are the future of human travel.")
script_text = generate_text(user_input, 1000)
script_path = os.path.join(temp_dir, "script.txt")
with open(script_path, 'w') as f:
    f.write(script_text)
logging.info("Generated text script.")

###########################################################################

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

# Load the music file (Assuming the music files are in the current directory)
audio = AudioSegment.from_mp3(f'music/{selected_file}')

music_path= audio



###########################################################################
# logging.info(f"Background music saved at {music_path}")

# Step 3: Generate Voice-Over Audio
voice_over_path = generate_audio(script_text, os.path.join(temp_dir, "voice_over.wav"))
logging.info(f"Voice-over audio saved at {voice_over_path}")

# Step 4: Transcribe Audio to Generate Subtitles
transcription_path = transcribe_audio(voice_over_path, temp_dir)
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
subtitles_path = os.path.join(temp_dir, "subtitles.json")
images_paths = [os.path.join(temp_dir, f"image_{i}.png") for i in range(0, 5)]  # Assuming 5 images
output_video_path = os.path.join(temp_dir, "final_video.mp4")
assemble_video(images_paths, voice_over_path, music_path, subtitles_path, output_video_path)
logging.info(f"Final video saved at {output_video_path}")

# Move the final video to the current directory
final_output = os.path.join(os.getcwd(), "final_video.mp4")
shutil.move(output_video_path, final_output)
logging.info(f"Video moved to {final_output}")

logging.info("All done!")