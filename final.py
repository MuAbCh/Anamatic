import os
import logging
import random
import json
from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    CompositeAudioClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.audio.fx.all import volumex
from pydub import AudioSegment
from Step_1_Text_Generation import generate_text
from Step_3_Audio_Generation import generate_audio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of categories and keywords (for easy reference)
categories = {
    "Nature Landscapes": ["Serenity", "Vibrant", "Expanse", "Majestic"],
    "Urban Architecture": ["Skyline", "Symmetry", "Modern", "Monumental"],
    "Space Exploration": ["Cosmic", "Orbital", "Astronomical", "Expedition"],
    "Animals in the Wild": ["Predator", "Herd", "Exotic", "Natural Habitat"],
    "Abstract Art": ["Geometric", "Colorful", "Surreal", "Chaotic"],
    "Historical Events": ["Revolutionary", "Ancient", "Conflict", "Iconic"],
    "Fantasy Worlds": ["Mystical", "Legendary", "Castle", "Sorcery"],
    "Technology and Gadgets": ["Futuristic", "Interactive", "Automated", "Wearable"],
    "Cultural Festivals": ["Vibrant Costumes", "Fireworks", "Parades", "Rituals"],
    "Sports in Action": ["Dynamic", "Victory", "Endurance", "Teamwork"],
    "Programming Concepts": ["Algorithm", "Debugging", "Syntax", "Automation"]
}

# Flatten the list of keywords
keyword_list = [keyword for keywords in categories.values() for keyword in keywords]

image_data = []
video_data = []

def generate_image_metadata():
    for category, keywords in categories.items():
        for i in range(3):
            img_keywords = random.sample(keywords, 2)
            formatted_category = category.replace(' ', '_')
            image_data.append({
                "image_path": f"{formatted_category}/{formatted_category}_image_{i + 1}.png",
                "keywords": img_keywords
            })

def generate_video_metadata():
    for category, keywords in categories.items():
        vid_keywords = random.sample(keywords, 2)
        formatted_category = category.replace(' ', '_')
        video_data.append({
            "video_path": f"{formatted_category}/{formatted_category}_video_1.mp4",
            "keywords": vid_keywords
        })

generate_image_metadata()
generate_video_metadata()

def find_best_video_match(query_keywords):
    best_match = None
    highest_score = 0
    for video in video_data:
        match_count = len(set(query_keywords).intersection(video['keywords']))
        if match_count > highest_score:
            best_match = video
            highest_score = match_count
    return best_match or random.choice(video_data)

def find_best_match(query_keywords):
    best_match = None
    highest_score = 0
    for image in image_data:
        match_count = len(set(query_keywords).intersection(image['keywords']))
        if match_count > highest_score:
            best_match = image
            highest_score = match_count
    return best_match or random.choice(image_data)

def clean_keywords(assigned_keywords):
    joined_keywords = ' '.join(assigned_keywords)
    cleaned_keywords = [word for word in joined_keywords.split() if word in keyword_list]
    return cleaned_keywords

def assign_keywords_to_sentence(sentence, keyword_list, model="llama-3.1-70b-versatile", temperature=0.00):
    system_prompt = "You are tasked with assigning two relevant keywords from a predefined list to the given sentence."
    user_prompt = f"""Here is the list of available keywords: {', '.join(keyword_list)}. 
    Assign two keywords from this list that best describe the following sentence: "{sentence}"."""
    
    keywords = generate_text(system_prompt, user_prompt, self_model=model, self_temperature=temperature)
    return clean_keywords(keywords.split(", "))

def process_script_cleaned(script, keyword_list):
    sentences = script.split("\n")
    sentence_keywords = {}
    for sentence in sentences:
        assigned_keywords = assign_keywords_to_sentence(sentence, keyword_list)
        sentence_keywords[sentence] = assigned_keywords
    return sentence_keywords

def save_script_and_metadata(script, sentence_keywords_cleaned, selected_images, concept):
    data = {
        "script": script,
        "sentence_keywords": sentence_keywords_cleaned,
        "selected_images": selected_images
    }
    file_name = f"{concept}_script_and_metadata.json"
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Script and metadata saved to {file_name}")

def assemble_video(image_paths, voice_over_path, music_path, output_path, concept):
    try:
        voice_over = AudioFileClip(voice_over_path)
        logging.info(f"Voice-over loaded: {voice_over_path}")
    except Exception as e:
        logging.error(f"Failed to load voice-over audio: {e}")
        return

    total_duration = voice_over.duration
    logging.info(f"Total duration: {total_duration}")

    clips = []
    clip_duration = total_duration / len(image_paths)
    
    for index, (sentence, path) in enumerate(image_paths.items()):
        if index == 0 or index == len(image_paths) - 1:
            try:
                clip = VideoFileClip(path).set_duration(clip_duration)
            except Exception as e:
                logging.error(f"Failed to load video clip: {e}")
                clip = ImageClip(path).set_duration(clip_duration)
        else:
            clip = ImageClip(path).set_duration(clip_duration)
        clips.append(clip)

    video = concatenate_videoclips(clips)
    video = video.set_duration(total_duration)

    try:
        music = AudioFileClip(music_path)
        logging.info(f"Background music loaded: {music_path}")
        
        if music.duration < total_duration:
            music = music.fx(volumex.audio_loop, duration=total_duration)
        else:
            music = music.subclip(0, total_duration)
        
        music = music.volumex(0.1)
    except Exception as e:
        logging.error(f"Failed to load or process background music: {e}")
        music = None

    if music:
        final_audio = CompositeAudioClip([voice_over, music])
    else:
        final_audio = voice_over

    final_video = video.set_audio(final_audio)

    try:
    # Trim the video to the length of the final audio
        if final_audio:
            final_audio_duration = final_audio.duration  # Get the duration of the final audio
            video = video.subclip(0, final_audio_duration)  # Trim the video to match the audio length

        # Write the final video to the specified output path
        video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            threads=4,
            preset='medium',
            bitrate="5000k"
        )
        logging.info(f"Final video saved at {output_path}")
    except Exception as e:
        logging.error(f"Failed to write the final video file: {e}")
        # finally:
    #     final_video.close()
    #     voice_over.close()
    #     if music:
    #         music.close()

def main():
    concept = input("Enter the topic you would like to make videos about: ").replace(" ", "_").lower()
    concept_dir = os.path.join(".", concept)
    os.makedirs(concept_dir, exist_ok=True)

    system_prompt = "You are the first step in a pipeline designed to create YouTube Shorts videos on a topic provided by the user. Your specific role is to generate a script that will later be paired with relevant visuals (images and videos)."
    user_prompt = f"I need you to write a script for a YouTube Short based on the topic: {concept}. The entire script must be at most 225 words long."

    script = generate_text(prompt_system=system_prompt, prompt_user=user_prompt)

    system_prompt = "Closely follow the User's instructions."
    user_prompt = f"Here is the current script: {script}\nRemove any headings that say 'Here is the script' etc. The script should start right away. I don't want the 'visual descriptions' in the script, the TTS model will speak the script as it is written. Also remove the seconds of how long the sentence is, we will calculate that on our own. Make sure that the script does not have any fancy formatting. Each new line should be separated by just one \\n, no fancy spacings etc etc"

    script = generate_text(prompt_system=system_prompt, prompt_user=user_prompt)

    sentence_keywords_cleaned = process_script_cleaned(script, keyword_list)

    selected_images = {}
    for index, (sentence, keywords) in enumerate(sentence_keywords_cleaned.items()):
        if index == 0 or index == len(sentence_keywords_cleaned) - 1:
            best_video = find_best_video_match(keywords)
            selected_images[sentence] = best_video['video_path']
        else:
            best_image = find_best_match(keywords)
            selected_images[sentence] = best_image['image_path']

    save_script_and_metadata(script, sentence_keywords_cleaned, selected_images, concept)

    voice_over_path = generate_audio(text=script, person=1, dir=concept)

    music_files = [
        "BRAZILIAN DANÇA PHONK.mp3",
        "Double Life (From _Despicable Me 4_).mp3",
        "edit another piano.mp3",
        "edit_audio_2.mp3",
        "edit_good_ending.mp3",
        "edit_piano.mp3"
    ]
    selected_file = random.choice(music_files)
    music_file_path = os.path.abspath(f'music/{selected_file}')

    output_path = os.path.join(concept, f"{concept}_final.mp4")
    assemble_video(selected_images, voice_over_path, music_file_path, output_path, concept)

    final_output = os.path.join(os.getcwd(), f"{concept}_final.mp4")
    if os.path.exists(output_path):
        os.replace(output_path, final_output)
        logging.info(f"Video moved to {final_output}")
    else:
        logging.error(f"Final video not found at {output_path}")

    logging.info("All done!")

if __name__ == "__main__":
    main()