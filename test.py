import random
from pydub import AudioSegment
from moviepy.editor import AudioFileClip
from moviepy.editor import afx
import os

def he(music_file_path, total_duration=60):
    def increase_volume_at_start(audio, duration=500):  # 500 ms = 0.5 seconds
        initial = audio[:duration]
        rest = audio[duration:]
        initial = initial + 6  # Increase volume by 6 dB
        return initial + rest
    
    # Add Background Music and loop it to match the video duration
    try:
        # Load the audio file again as an AudioSegment
        audio = AudioSegment.from_mp3(music_file_path)
        
        # Determine if the file name contains "edit"
        if "edit" in os.path.basename(music_file_path):
            # Keep only the last x seconds (where x = total_duration)
            start_time = len(audio) - (total_duration * 1000)  # convert seconds to milliseconds
            edited_audio = audio[start_time:]
            
            # Increase the volume at the start
            edited_audio = increase_volume_at_start(edited_audio, 1500)
        else:
            # For non-edit files, keep only the first seconds (based on video length)
            edited_audio = audio[:total_duration * 1000]

        # Export the edited audio to a temporary file
        edited_file_path = "edited_music.mp3"
        edited_audio.export(edited_file_path, format="mp3")
        
        # Use the edited file with AudioFileClip
        music = AudioFileClip(edited_file_path).volumex(0.1)  # Lower volume for background music
        # Loop the music to ensure it covers the entire video duration using afx.audio_loop
        music = music.fx(afx.audio_loop, duration=total_duration)
        
        return music
    
    except Exception as e:
        print(f"Error processing music: {e}")
        return None

music_files = [
    "BRAZILIAN DANÇA PHONK.mp3",
    "Double Life (From _Despicable Me 4_).mp3",
    "edit another piano.mp3",
    "edit_audio_2.mp3",
    "edit_good_ending.mp3",
    "edit_piano.mp3"
]

# Set total_duration (replace with actual video length in seconds)
total_duration = 60  # Example video length

# Choose a random music file
selected_file = random.choice(music_files)

# Set the full path to the selected file
music_file_path = f'music/{selected_file}'

# Process the music with the he() function
music = he(music_file_path, total_duration)
