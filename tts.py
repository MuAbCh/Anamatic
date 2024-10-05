from Step_3_Audio_Generation import generate_audio
from Step_4_Transcript_Generation import transcribe_audio, make_file_path



txt = """Imagine walking on the moon's surface, surrounded by craters and vast, barren landscapes.

The moon is Earth's only natural satellite, formed about 4.5 billion years ago from debris left over after a massive collision.

NASA's Apollo missions successfully landed astronauts on the moon six times between 1969 and 1972.

One of the most significant discoveries on the moon was water ice, found in permanently shadowed craters near the lunar poles.

The moon's surface is also home to 'dark side' craters, which are hidden from Earth due to the moon's synchronous rotation.

Recent missions, such as China's Chang'e 4, have explored the moon's far side and discovered new insights into its geology and composition.

As we continue to explore the moon and space, we may uncover even more secrets about our universe and its mysteries.

The moon remains an essential stepping stone for humanity's journey into space, driving innovation and inspiring future generations.

Join us as we continue to explore the wonders of the moon and the vastness of space."""


concept = "new_concept"
import os

import os

# Assuming this is how the audio file is generated
full_audio_path = generate_audio(text=txt, person=1, dir=concept)
# Example: full_audio_path might be something like 'new_concept/new_audio.wav'

# Join the path with the current directory for consistent path handling
full_audio_path = os.path.join(os.getcwd(), full_audio_path)

# Get just the file name (without the path) for printing purposes
audio_file = os.path.basename(full_audio_path)
print(f"Generated the Audio file: {audio_file}")

# Generate the subtitle file path
subtitle_file = make_file_path(concept, 'srt')  # Ensuring it's the full path

# Ensure you pass the full path when calling transcribe_audio
transcribe_audio(audio_file=full_audio_path, subtitle_file=subtitle_file, concept=concept)

print("Done")



