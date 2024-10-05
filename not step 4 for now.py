import os
import whisper
import whisperx
import torch
import ffmpeg
from whisperx.utils import get_writer
import re  # Add this import

def load_whisper_model(device = 'cpu', compute_type = 'float16'):
    whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)
  # Or "small", "medium", etc.
    return whisper_model

def update_ass_styles(input_file, output_file, new_styles):
    
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Define the pattern to find the [V4+ Styles] section
    style_pattern = re.compile(r'^Style:.*$')

    with open(output_file, 'w', encoding='utf-8') as file:
        in_styles_section = False
        for line in lines:
            if line.startswith('[V4+ Styles]'):
                in_styles_section = True
                file.write(line)
                continue
            if in_styles_section and line.startswith('['):
                in_styles_section = False

            if in_styles_section and style_pattern.match(line):
                style_name = line.split(',')[0].split(':')[1].strip()
                if style_name in new_styles:
                    style_parts = line.strip().split(',')
                    for key, value in new_styles[style_name].items():
                        style_parts[key] = value
                    new_line = ','.join(style_parts) + '\n'
                    file.write(new_line)
                else:
                    file.write(line)
            else:
                file.write(line)


def make_file_path(concept, extension):
 
    # Replace spaces with underscores in the concept name
    concept_dir = concept.replace(" ", "_")
    
    # Create the directory if it doesn't exist
    if not os.path.isdir(concept_dir):
        os.mkdir(concept_dir)
    
    # Generate the file path with the specified extension
    file_path = os.path.join(concept_dir, f'{concept_dir}.{extension}')
    
    # Create an empty file (or write a placeholder) if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write('')  # Writing an empty string to ensure the file is created
    
    return file_path



def transcribe_audio(audio_file_new, subtitle_file, whisper_model, style='Default'):
    """
    Transcribes an audio file into subtitles with word-by-word highlighting, aligns the transcription using WhisperX, 
    and updates the subtitles file with custom styles. Removes overlapping subtitles.

    Parameters:
    - audio_file_new (str): The path to the new audio file to be transcribed.
    - subtitle_file (str): The path where the generated subtitle file will be saved and updated.
    - whisper_model (Whisper Model Object): The Whisper model used for transcribing the audio.
    - style (str): The name of the style to be applied to the subtitles. Default is 'Default'.

    Returns:
    - None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16

    if os.path.exists(audio_file_new):
        # Load audio using WhisperX
        audio = whisperx.load_audio(audio_file_new)

        # Transcribe using Whisper
        result = whisper_model.transcribe(audio, batch_size=batch_size)
        language_code = result["language"]

        # Load alignment model from WhisperX
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)

        # Align transcriptions using WhisperX
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        result['language'] = language_code

        # Write the result into an SRT file
        srt_writer = get_writer("srt", ".")

        srt_writer(
            result,
            audio_file_new,
            {"max_line_width": 15, "max_line_count": 1, "highlight_words": True}
        )

        print(f"Subtitles saved to {subtitle_file}")

        # Optionally apply styles and remove overlaps
        update_ass_styles(subtitle_file, subtitle_file, style)  # Assuming `update_ass_styles` is defined
        remove_overlapping_subtitles(subtitle_file)  # Assuming `remove_overlapping_subtitles` is defined
    else:
        print(f"Audio file {audio_file_new} does not exist!")

# We gotta make these files before we store the stuff.
# Maybe pass in the name of the files or smth, and the current dir:
concept = ''
    # making the files we need:
audio_file = make_file_path(concept, 'wav')          # Path to the audio file
subtitle_file = make_file_path(concept, 'ass')       # Path to the ASS subtitle file
output_file = make_file_path(concept, 'mp4')  
transcribe_audio()












































# new_styles = {
#     "Default": {
#         1: "Arial",             # Fontname
#         2: "12",                # Fontsize
#         3: "&H00FF00FF",        # PrimaryColour (green)
#         4: "&H000000FF",        # SecondaryColour
#         5: "&H00000000",        # OutlineColour
#         6: "&H64000000",        # BackColour
#         16: "2",                # Outline (thickness)
#         17: "1"                 # Shadow (depth)
#     },
#     "Bold": {
#         1: "Arial Black",       # Fontname
#         2: "14",                # Fontsize
#         3: "&H0000FFFF",        # PrimaryColour (red)
#         4: "&H00FFFFFF",        # SecondaryColour
#         5: "&H00000000",        # OutlineColour
#         6: "&H64000000",        # BackColour
#         16: "3",                # Outline (thickness)
#         17: "2",                # Shadow (depth)
#         21: "1"                 # Bold
#     },
#     "Italic": {
#         1: "Georgia",           # Fontname
#         2: "12",                # Fontsize
#         3: "&H00FF00FF",        # PrimaryColour (green)
#         4: "&H000000FF",        # SecondaryColour
#         5: "&H00000000",        # OutlineColour
#         6: "&H64000000",        # BackColour
#         16: "1",                # Outline (thickness)
#         17: "0",                # Shadow (depth)
#         21: "1"                 # Italic
#     },
#     "Highlight": {
#         1: "Arial",             # Fontname
#         2: "12",                # Fontsize
#         3: "&H00FFFF00",        # PrimaryColour (yellow)
#         4: "&H000000FF",        # SecondaryColour
#         5: "&H00000000",        # OutlineColour
#         6: "&H64000000",        # BackColour
#         16: "4",                # Outline (thickness)
#         17: "0"                 # Shadow (depth)
#     },
#     "Shadowed": {
#         1: "Verdana",           # Fontname
#         2: "12",                # Fontsize
#         3: "&H00FFFFFF",        # PrimaryColour (white)
#         4: "&H000000FF",        # SecondaryColour
#         5: "&H00000000",        # OutlineColour
#         6: "&H64000000",        # BackColour
#         16: "1",                # Outline (thickness)
#         17: "4"                 # Shadow (depth)
#     },
#     "OutlineThick": {
#         1: "Tahoma",            # Fontname
#         2: "14",                # Fontsize
#         3: "&H00000000",        # PrimaryColour (black)
#         4: "&H00FFFFFF",        # SecondaryColour (white)
#         5: "&H00FF0000",        # OutlineColour (red)
#         6: "&H64000000",        # BackColour
#         16: "6",                # Outline (thickness)
#         17: "0"                 # Shadow (depth)
#     },
#     "Glow": {
#         1: "Calibri",           # Fontname
#         2: "13",                # Fontsize
#         3: "&H00FFFFFF",        # PrimaryColour (white)
#         4: "&H0000FFFF",        # SecondaryColour (blue)
#         5: "&H00000000",        # OutlineColour (black)
#         6: "&H64FFFF00",        # BackColour (yellow glow)
#         16: "2",                # Outline (thickness)
#         17: "3"                 # Shadow (depth)
#     },
#     "Mono": {
#         1: "Courier New",       # Fontname
#         2: "11",                # Fontsize
#         3: "&H00FFFFFF",        # PrimaryColour (white)
#         4: "&H000000FF",        # SecondaryColour
#         5: "&H00000000",        # OutlineColour
#         6: "&H64000000",        # BackColour
#         16: "0",                # Outline (thickness)
#         17: "0"                 # Shadow (depth)
#     },
#     "Comic": {
#         1: "Comic Sans MS",     # Fontname
#         2: "16",                # Fontsize
#         3: "&H00FFA500",        # PrimaryColour (orange)
#         4: "&H000000FF",        # SecondaryColour
#         5: "&H00000000",        # OutlineColour
#         6: "&H64000000",        # BackColour
#         16: "2",                # Outline (thickness)
#         17: "1"                 # Shadow (depth)
#     },
#     "Fancy": {
#         1: "Times New Roman",   # Fontname
#         2: "18",                # Fontsize
#         3: "&H00FFC0CB",        # PrimaryColour (pink)
#         4: "&H000000FF",        # SecondaryColour
#         5: "&H00000000",        # OutlineColour
#         6: "&H64FFFFFF",        # BackColour (white)
#         16: "1",                # Outline (thickness)
#         17: "2"                 # Shadow (depth)
#     },
#     "Retro": {
#         1: "Lucida Console",    # Fontname
#         2: "15",                # Fontsize
#         3: "&H00FFD700",        # PrimaryColour (gold)
#         4: "&H000000FF",        # SecondaryColour
#         5: "&H00000000",        # OutlineColour
#         6: "&H64FF0000",        # BackColour (red)
#         16: "2",                # Outline (thickness)
#         17: "1"                 # Shadow (depth)
#     },
#     "Minimalist": {
#         1: "Helvetica",         # Fontname
#         2: "10",                # Fontsize
#         3: "&H00FFFFFF",        # PrimaryColour (white)
#         4: "&H000000FF",        # SecondaryColour
#         5: "&H00000000",        # OutlineColour
#         6: "&H00000000",        # BackColour
#         16: "0",                # Outline (thickness)
#         17: "0"                 # Shadow (depth)
#     }
# }