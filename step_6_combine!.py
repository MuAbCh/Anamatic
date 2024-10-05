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


from Step_1_Text_Generation import generate_text
from Step_2_Music_Generation import generate_music_file
from Step_3_Audio_Generation import gen
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1 code ###########################################################

<<<<<<< HEAD


=======
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

>>>>>>> 90c9a8a904a40074b904dfcb6da0da3639fbffb8

# Step 2 code ###########################################################



# Step 3 Audio Generation ###########################################################
# Original Working `synthesize_text_to_audio` Function
def synthesize_text_to_audio(
    text: str,
    output_path: str = "output/audio.wav",
    matcha_checkpoint: Path = None,
    hifigan_checkpoint: Path = None,
    n_timesteps: int = 10,
    length_scale: float = 1.0,
    temperature: float = 0.667,
    device: torch.device = None
):
    """
    Synthesizes speech from the input text and saves it as a WAV file.

    Parameters:
    - text (str): The input text to synthesize.
    - output_path (str): Path to save the output WAV file.
    - matcha_checkpoint (Path): Path to the Matcha-TTS checkpoint. Defaults to user data directory.
    - hifigan_checkpoint (Path): Path to the HiFi-GAN checkpoint. Defaults to user data directory.
    - n_timesteps (int): Number of ODE solver steps.
    - length_scale (float): Changes to the speaking rate.
    - temperature (float): Sampling temperature.
    - device (torch.device): Device to run the models on. Defaults to CUDA if available.

    Returns:
    - dict: Contains the synthesized waveform and related metadata.
    """
    import numpy as np
    import soundfile as sf
    from matcha.hifigan.config import v1
    from matcha.hifigan.denoiser import Denoiser
    from matcha.hifigan.env import AttrDict
    from matcha.hifigan.models import Generator as HiFiGAN
    from matcha.models.matcha_tts import MatchaTTS
    from matcha.text import sequence_to_text, text_to_sequence
    from matcha.utils.utils import get_user_data_dir, intersperse

    # Initialize device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set default checkpoint paths if not provided
    if matcha_checkpoint is None:
        matcha_checkpoint = get_user_data_dir() / "matcha_ljspeech.ckpt"
    if hifigan_checkpoint is None:
        hifigan_checkpoint = get_user_data_dir() / "hifigan_T2_v1"

    # Initialize models only once
    if not hasattr(synthesize_text_to_audio, "model"):
        # Load Matcha-TTS model
        print("Loading Matcha-TTS model...")
        synthesize_text_to_audio.model = MatchaTTS.load_from_checkpoint(
            matcha_checkpoint, map_location=device
        ).to(device)
        synthesize_text_to_audio.model.eval()
        print("Matcha-TTS model loaded.")

        # Load HiFi-GAN vocoder
        print("Loading HiFi-GAN vocoder...")
        h = AttrDict(v1)
        synthesize_text_to_audio.vocoder = HiFiGAN(h).to(device)
        synthesize_text_to_audio.vocoder.load_state_dict(
            torch.load(hifigan_checkpoint, map_location=device)["generator"]
        )
        synthesize_text_to_audio.vocoder.eval()
        synthesize_text_to_audio.vocoder.remove_weight_norm()
        print("HiFi-GAN vocoder loaded.")

        # Initialize Denoiser
        synthesize_text_to_audio.denoiser = Denoiser(synthesize_text_to_audio.vocoder, mode="zeros")

    model = synthesize_text_to_audio.model
    vocoder = synthesize_text_to_audio.vocoder
    denoiser = synthesize_text_to_audio.denoiser

    # Define helper functions within the main function

    @torch.inference_mode()
    def process_text(text_input: str):
        x = torch.tensor(
            intersperse(text_to_sequence(text_input, ['english_cleaners2'])[0], 0),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)
        x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
        x_phones = sequence_to_text(x.squeeze(0).tolist())
        return {
            'x_orig': text_input,
            'x': x,
            'x_lengths': x_lengths,
            'x_phones': x_phones
        }

    @torch.inference_mode()
    def synthesise(text_processed):
        start_time = dt.datetime.now()
        output = model.synthesise(
            text_processed['x'],
            text_processed['x_lengths'],
            n_timesteps=n_timesteps,
            temperature=temperature,
            spks=None,  # Modify if speaker embeddings are used
            length_scale=length_scale
        )
        output.update({'start_t': start_time, **text_processed})
        return output

    @torch.inference_mode()
    def to_waveform(mel_spec):
        audio = vocoder(mel_spec).clamp(-1, 1)
        audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
        return audio.numpy()

    def save_audio(waveform, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, waveform, 22050, subtype='PCM_24')
        print(f"Audio saved to {path}")

    # Process the input text
    text_processed = process_text(text)

    # Synthesize the mel spectrogram
    output = synthesise(text_processed)

    # Convert mel spectrogram to waveform
    waveform = to_waveform(output['mel'])

    # Save the waveform to the specified output path
    save_audio(waveform, output_path)

    # Optionally, return the waveform and other details
    return output_path


def transcribe_audio(audio_path, temp_dir):
    """
    Transcribes the audio file to generate subtitles with timestamps.

    Parameters:
    - audio_path (str): Path to the audio file.
    - temp_dir (str): Path to the temporary directory.

    Returns:
    - str: Path to the transcription text file.
    """
    transcription_fn = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
        device=0 if torch.cuda.is_available() else -1,
    )

    audio_input, sr = sf.read(audio_path)
    inputs = {
        "raw": audio_input,
        "sampling_rate": sr,
    }

    transcription_chunks = transcription_fn(
        inputs,
        batch_size=8,
        return_timestamps=True
    ).get("chunks", [])

    formatted_transcription = ""
    for segment in transcription_chunks:
        text = segment.get("text", "").strip()
        start, end = segment.get("timestamp", (0.0, 0.0))
        start_rounded = round(start, 2)
        end_rounded = round(end, 2) if end else round(start + 5.0, 2)
        formatted_transcription += f"{text} | start: {start_rounded} | end: {end_rounded}\n"

    transcription_path = os.path.join(temp_dir, "transcription.txt")
    with open(transcription_path, 'w') as f:
        f.write(formatted_transcription)

    del transcription_fn, audio_input, sr, inputs, transcription_chunks, formatted_transcription
    gc.collect()

    return transcription_path


def generate_image_descriptions(transcription, temp_dir):
    """
    Generates image descriptions based on the transcription.

    Parameters:
    - transcription (str): Path to the transcription text file.
    - temp_dir (str): Path to the temporary directory.

    Returns:
    - str: Path to the JSON file containing image descriptions.
    """
    script = """
You are given a transcript of a short video with timestamps.
You are in charge of making a list of pictures that will be used to create a video.
The video will be a slideshow of the pictures.
The pictures should be relevant to the text.
Make sure to include how long each picture should be displayed as well as the description of the picture.
Make only 5 images.

Example JSON output:
{"images": [{"description": "A picture of a cat", "start": 1, "end": 3}, {"description": "A picture of a dog", "start": 3, "end": 5}]}  
"""

    with open(transcription, 'r') as f:
        transcript_text = f.read()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": script},
            {"role": "user", "content": transcript_text}
        ],
        response_format={ "type": "json_object" }
    )

    try:
        images = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from OpenAI response: {e}")
        images = []

    images_path = os.path.join(temp_dir, "images.json")
    with open(images_path, 'w') as f:
        json.dump(images, f, indent=4)

    del response, transcript_text, images
    gc.collect()

    return images_path


def setup_sdxl_base_model(model_dir, device="CPU", compress_weights=True):
    """
    Sets up the Stable Diffusion XL Base model optimized with OpenVINO.

    Parameters:
    - model_dir (str): Directory to save the converted model.
    - device (str): Inference device ('CPU', 'GPU.0', etc.).
    - compress_weights (bool): Whether to apply 8-bit weight compression.

    Returns:
    - OVStableDiffusionXLPipeline: The optimized pipeline.
    """
    quantization_config = {"bits": 8} if compress_weights else None

    if not Path(model_dir).exists():
        pipeline = OVStableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            compile=False,
            device=device,
            quantization_config=quantization_config
        )
        pipeline.half()
        pipeline.save_pretrained(model_dir)
        pipeline.compile()
    else:
        pipeline = OVStableDiffusionXLPipeline.from_pretrained(
            model_dir,
            device=device
        )
    gc.collect()
    return pipeline


def generate_images(images_json_path, base_pipeline, temp_dir):
    """
    Generates images based on the provided descriptions.

    Parameters:
    - images_json_path (str): Path to the JSON file containing image descriptions.
    - base_pipeline (OVStableDiffusionXLPipeline): The image generation pipeline.
    - temp_dir (str): Path to the temporary directory.

    Returns:
    - list: List of paths to the generated images.
    """
    with open(images_json_path, 'r') as f:
        images = json.load(f)

    generated_images = []
    print(images)
    for idx, image_info in enumerate(images['images']):
        description = image_info["description"]
        print(f"Generating image {idx+1}: {description}")
        image = base_pipeline(
            prompt=description,
            num_inference_steps=15,
            height=512,
            width=512,
            generator=np.random.RandomState(42)
        ).images[0]
        image_path = os.path.join(temp_dir, f"image_{idx}.png")
        image.save(image_path)
        generated_images.append(image_path)

    del base_pipeline, images, image
    gc.collect()

    return generated_images


def generate_subtitles(transcription_path, temp_dir):
    """
    Formats the transcription into a JSON file suitable for subtitles.

    Parameters:
    - transcription_path (str): Path to the transcription text file.
    - temp_dir (str): Path to the temporary directory.

    Returns:
    - str: Path to the JSON file containing subtitles.
    """
    subtitles = []
    with open(transcription_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split('|')
        if len(parts) != 3:
            continue
        text = parts[0].strip()
        start = float(parts[1].replace('start:', '').strip())
        end = float(parts[2].replace('end:', '').strip())
        subtitles.append({"text": text, "start": start, "end": end})

    subtitles_path = os.path.join(temp_dir, "subtitles.json")
    with open(subtitles_path, 'w') as f:
        json.dump(subtitles, f, indent=4)

    return subtitles_path


def assemble_video(image_paths, voice_over_path, music_path, subtitles_path, output_path):
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

    # Load subtitles
    try:
        with open(subtitles_path, 'r') as f:
            subtitles = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load subtitles: {e}")
        subtitles = []

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

    # Determine the total duration of the video based on subtitles and extended voice-over
    if subtitles:
        subtitles_max_end = max(sub['end'] for sub in subtitles)
    else:
        subtitles_max_end = 0

    if voice_over_extended:
        voice_over_duration = voice_over_extended.duration
    else:
        voice_over_duration = 0

    # Calculate total_duration without buffer
    total_duration = max(subtitles_max_end, voice_over_duration)

    logging.debug(f"Subtitles Max End: {subtitles_max_end} seconds")
    logging.debug(f"Voice-Over Extended Duration: {voice_over_duration} seconds")
    logging.debug(f"Total Duration: {total_duration} seconds")

    # Calculate the duration each image will be displayed
    num_images = len(image_paths)
    if num_images > 0:
        # Adjust image_duration to account for crossfades
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

    # Add Background Music and loop it to match the video duration
    try:
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

    # Add Subtitles
    for subtitle in subtitles:
        try:
            txt_clip = TextClip(
                subtitle["text"],
                fontsize=24,
                color='white',
                bg_color='black',
                method='caption',
                size=(video.w * 0.8, None),
                align='center'
            )
            txt_clip = txt_clip.set_start(subtitle["start"]).set_end(subtitle["end"]).set_position(('center', 'bottom'))
            video = CompositeVideoClip([video, txt_clip])
        except Exception as e:
            logging.error(f"Failed to add subtitle '{subtitle}': {e}")

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

# with tempfile.TemporaryDirectory() as temp_dir:
temp_dir = "temp"
logging.info(f"Using temporary directory: {temp_dir}")

# Step 1: Generate Text Script
prompt = """\n\n
Your task is to create a 30 second engaging and educational TikTok script based on the following sentence:

{input_sentence}

Expand on this sentence to create an interesting and educational script that most people might not know about.
The TikTok should incorporate an engaging story or example related to the sentence.
Do not include any emojis or hashtags in the script.
The script should be only spoken text, no extra text like [Cut] or [Music].
The script should sound passionate, excited, and happy.

Script:
"""
user_input = prompt.format(input_sentence="Spaceships are the future of human travel.")
script_text = generate_text(user_input, 1000)
script_path = os.path.join(temp_dir, "script.txt")
with open(script_path, 'w') as f:
    f.write(script_text)
logging.info("Generated text script.")

# Step 2: Generate Music Description
music_description = generate_music_description(script_text, generate_text, temp_dir)
logging.info(f"Music Description: {music_description}")

# Step 3: Generate Background Music
music_path = generate_music(music_description, temp_dir)
logging.info(f"Background music saved at {music_path}")

# Step 4: Generate Voice-Over Audio
voice_over_path = synthesize_text_to_audio(script_text, os.path.join(temp_dir, "voice_over.wav"))
logging.info(f"Voice-over audio saved at {voice_over_path}")

# Step 5: Transcribe Audio to Generate Subtitles
transcription_path = transcribe_audio(voice_over_path, temp_dir)
logging.info(f"Transcription saved at {transcription_path}")

# Step 6: Generate Image Descriptions
images_json_path = generate_image_descriptions(transcription_path, temp_dir)
logging.info(f"Image descriptions saved at {images_json_path}")

# Step 7: Generate Images
model_dir = os.path.join(temp_dir, "openvino-sd-xl-base-1.0")
base_pipeline = setup_sdxl_base_model(model_dir=model_dir, device="CPU")
images_paths = generate_images(images_json_path, base_pipeline, temp_dir)
logging.info(f"Generated images: {images_paths}")

# Step 8: Generate Subtitles File
subtitles_path = generate_subtitles(transcription_path, temp_dir)
logging.info(f"Subtitles saved at {subtitles_path}")


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