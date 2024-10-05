import random
from pydub import AudioSegment
from moviepy.editor import AudioFileClip
from moviepy.editor import afx
import os



import re

import re
import logging

import re
import logging

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







srt_file = "new_srt.srt"
lines_to_highlight_dict = [
    "And then we initialize a set to keep track",
    "visited vertices and a stack with the",
    "starting vertex and."
]



highlight_times = update_highlight_times(srt_file, lines_to_highlight_dict)
print(highlight_times)






































# def he(music_file_path, total_duration=60):
#     def increase_volume_at_start(audio, duration=500):  # 500 ms = 0.5 seconds
#         initial = audio[:duration]
#         rest = audio[duration:]
#         initial = initial + 6  # Increase volume by 6 dB
#         return initial + rest
    
#     # Add Background Music and loop it to match the video duration
#     try:
#         # Load the audio file again as an AudioSegment
#         audio = AudioSegment.from_mp3(music_file_path)
        
#         # Determine if the file name contains "edit"
#         if "edit" in os.path.basename(music_file_path):
#             # Keep only the last x seconds (where x = total_duration)
#             start_time = len(audio) - (total_duration * 1000)  # convert seconds to milliseconds
#             edited_audio = audio[start_time:]
            
#             # Increase the volume at the start
#             edited_audio = increase_volume_at_start(edited_audio, 1500)
#         else:
#             # For non-edit files, keep only the first seconds (based on video length)
#             edited_audio = audio[:total_duration * 1000]

#         # Export the edited audio to a temporary file
#         edited_file_path = "edited_music.mp3"
#         edited_audio.export(edited_file_path, format="mp3")
        
#         # Use the edited file with AudioFileClip
#         music = AudioFileClip(edited_file_path).volumex(0.1)  # Lower volume for background music
#         # Loop the music to ensure it covers the entire video duration using afx.audio_loop
#         music = music.fx(afx.audio_loop, duration=total_duration)
        
#         return music
    
#     except Exception as e:
#         print(f"Error processing music: {e}")
#         return None

# music_files = [
#     "BRAZILIAN DANÇA PHONK.mp3",
#     "Double Life (From _Despicable Me 4_).mp3",
#     "edit another piano.mp3",
#     "edit_audio_2.mp3",
#     "edit_good_ending.mp3",
#     "edit_piano.mp3"
# ]

# # Set total_duration (replace with actual video length in seconds)
# total_duration = 60  # Example video length

# # Choose a random music file
# selected_file = random.choice(music_files)

# # Set the full path to the selected file
# music_file_path = f'music/{selected_file}'

# # Process the music with the he() function
# music = he(music_file_path, total_duration)
