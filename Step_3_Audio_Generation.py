import os
import datetime as dt
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

# Import necessary modules from Matcha and HiFi-GAN
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.model import denormalize
from matcha.utils.utils import get_user_data_dir, intersperse

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
    """
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
        synthesize_text_to_audio.model = synthesize_text_to_audio.model.to(device)
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
    return {
        'text': output['x_orig'],
        'phonetic': output['x_phones's],
        'waveform': waveform,
        'rtf': None  # Real-Time Factor can be computed if needed
    }


result = synthesize_text_to_audio("Imagine a world where humanity has finally reached the stars, and we\'re not just talking about any old spacecraft, but the most advanced, cutting-edge, and sustainable vessels that are changing the game. Spaceships are not just a luxury, they\'re a necessity. They\'re the key to unlocking new frontiers, new discoveries, and new possibilities for humanity. But did you know that the first spaceship was actually a hot air balloon? Yes, you heard that right! In 1783, French inventor Montgolfier created the first successful hot air balloon, which carried a group of 20 people to the skies. It was a groundbreaking achievement that paved the way for the development of modern space travel. Fast forward to today, and we have reusable rockets, advanced propulsion systems, and even private space companies like SpaceX and Blue Origin pushing the boundaries of what\'s possible. But what\'s even more exciting is that we\'re not just talking about the technology, we\'re talking about the people, the communities, and the cultures that are being shaped by space exploration. From the astronauts who are pushing the limits of human endurance to the scientists who are unlocking the secrets of the universe, space travel is not just a dream, it\'s a reality that\'s changing our world. So let\'s get ready to blast off into the unknown, and explore the infinite possibilities that await us in the cosmos!")