from google.cloud import texttospeech
import os
import random
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "text2speechAPI.json"

client = texttospeech.TextToSpeechClient()

def generate_audio(text, person = 1, dir):
    random_number = person

    if (random_number == 1):
        voice = "en-US-Journey-F"
    elif (random_number == 2):
        voice = "en-US-Journey-O"
    else:
        voice = "en-US-Journey-O"

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    fileName = f"{dir}/output_{person}.mp3"

    with open(fileName, "wb") as out:
        out.write(response.audio_content)   
        print("Audio content written to output.mp3")

    return fileName
















# from pathlib import Path
# from openai import OpenAI
# import os
# import random
# from dotenv import load_dotenv

# load_dotenv(".env")
# OpenAI_API_KEY = os.getenv("OpenAI_API_KEY")
# client = OpenAI(api_key=OpenAI_API_KEY)

# random_number = random.randint(1, 3)

# if (random_number == 1):
#     voice = "alloy"
# elif (random_number == 2):
#     voice = "fable"
# else:
#     voice = "nova"

# speech_file_path = Path(__file__).parent / "speech.mp3"
# response = client.audio.speech.create(
#   model="tts-1",
#   voice=voice,
#   input="Today is a wonderful day to build something people love!"
# )

# response.stream_to_file(speech_file_path)