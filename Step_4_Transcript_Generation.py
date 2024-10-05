import whisperx
import ffmpeg
import os
import subprocess
from whisperx.utils import get_writer
import torch
import re



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


def remove_overlapping_subtitles(file_path):
   
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    cleaned_lines = []
    subtitle_pattern = re.compile(r"Dialogue:.*")
    timestamps = set()

    for line in lines:
        match = subtitle_pattern.match(line)
        if match:
            parts = line.split(',')
            start_time = parts[1].strip()
            end_time = parts[2].strip()

            if start_time in timestamps:
                continue
            timestamps.add(start_time)

        cleaned_lines.append(line)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(cleaned_lines)

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


def transcribe_audio(audio_file, subtitle_file,concept, style='Default'):

    batch_size = 16
    device = "cuda"
    whisper_model = whisperx.load_model("medium", device)
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)



    import ipdb; ipdb.set_trace()
    if os.path.exists(audio_file):
        audio = whisperx.load_audio(audio_file)
        result = whisper_model.transcribe(audio, batch_size=batch_size)
        language_code=result["language"]


        
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        result['language'] = language_code
        vtt_writer = get_writer("srt", f'{concept}/')
        vtt_writer(
            result,
            subtitle_file, #changed this from audio_file
            {"max_line_width": 15, "max_line_count": 1, "highlight_words": True},
        )


        file_name = f'{concept.replace(" ","_")}_new.srt'


        try:
            # joins the input file is the one being read and processed. the file_name is the one being outputted.

            ffmpeg.input(subtitle_file).output(file_name).overwrite_output().run()
        
        except ffmpeg.Error as e:
            # print(f"An error occurred: {e.stderr.decode()}")
            print("error")

        # (ffmpeg.input(os.path.join(concept.replace(" ","_"),file_name)).output(subtitle_file).run())

        # Usage
        # ass_name = os.path.join(concept.replace(" ","_"),f"{concept.replace(" ","_")}.ass")
        # we define the styles here:
        print(f"This is where the subtitle is file now: {subtitle_file} ")
        new_styles = {
    "Default": {
        1: "Arial",             # Fontname
        2: "12",                # Fontsize
        3: "&H00FF00FF",        # PrimaryColour (green)
        4: "&H000000FF",        # SecondaryColour
        5: "&H00000000",        # OutlineColour
        6: "&H64000000",        # BackColour
        16: "2",                # Outline (thickness)
        17: "1"                 # Shadow (depth)
    },
}
        update_ass_styles(subtitle_file, subtitle_file, new_styles)
        remove_overlapping_subtitles(subtitle_file)
        return subtitle_file

# # THIS IS HOW YOU CALL THE FUNCTION
# if __name__ == '__main__':
#     # concept = "not test" # FROM THE USER
#     concept = "new_concept"
#     concept = concept.replace(" ","_")
#     # audio_file = make_file_path(concept, 'wav')
#     audio_file = 'new_srt.wav'
#     subtitle_file = make_file_path(concept, 'srt')       # Path to the ASS subtitle file
#     # output_file = make_file_path(concept, 'mp4')         # Path to the output video file
#     transcribe_audio(audio_file=audio_file, subtitle_file=subtitle_file, concept=concept)