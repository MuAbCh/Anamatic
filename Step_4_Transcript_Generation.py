import torch
from transformers import pipeline
import soundfile as sf
from tqdm import tqdm
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transcribe_wav(wav_path, model_name="openai/whisper-tiny.en", device=None, task='transcribe', batch_size=8):
    """
    Transcribe a WAV audio file using Whisper with Hugging Face pipeline and return a formatted string.
    
    Parameters:
        wav_path (str or Path): Path to the input WAV audio file.
        model_name (str): Hugging Face model name.
        device (str or int, optional): Device to run the model on ('cpu', 'cuda:0', etc.). Defaults to automatic selection.
        task (str): Task type ('transcribe' or 'translate').
        batch_size (int): Batch size for processing.
    
    Returns:
        str: Formatted transcription string with timestamps.
    """
    try:
        # Determine device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        logging.info(f"Using device: {device}")

        # Initialize the pipeline
        logging.info(f"Loading model '{model_name}' for task '{task}'.")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device=device,
        )

        # Load the WAV file
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        
        logging.info(f"Loading audio file: {wav_path}")
        audio_input, sr = sf.read(wav_path)
        inputs = {
            "raw": audio_input,
            "sampling_rate": sr,
        }

        # Perform transcription with timestamps
        logging.info("Running transcription...")
        transcription_chunks = pipe(
            inputs,
            batch_size=batch_size,
            return_timestamps=True
        ).get("chunks", [])

        logging.info("Transcription completed.")

        # Format transcription into desired string
        formatted_transcription = format_transcription(transcription_chunks)

        return formatted_transcription

    except FileNotFoundError as fnf_error:
        logging.error(f"Error: {fnf_error}")
        return ""
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return ""

def format_transcription(transcription_chunks):
    """
    Format transcription chunks into the desired string format.

    Each line contains:
    Transcribed text | start: X | end: Y

    Parameters:
        transcription_chunks (list): List of transcription chunks with timestamps.

    Returns:
        str: Formatted transcription string.
    """
    formatted_output = ""
    for segment in tqdm(transcription_chunks, desc="Formatting transcription"):
        text = segment.get("text", "").strip()
        start, end = segment.get("timestamp", (0.0, 0.0))
        # Round the timestamps to two decimal places
        start_rounded = round(start, 2)
        end_rounded = round(end, 2) if end else round(start + 5.0, 2)  # Default to start + 5.0 if end is None
        formatted_output += f"{text} | start: {start_rounded} | end: {end_rounded}\n"
    return formatted_output



# Example usage:
transcription_text = transcribe_wav("output/audio.wav")
print(transcription_text)