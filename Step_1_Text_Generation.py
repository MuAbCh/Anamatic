import openvino_genai as ov_genai

def generate_text(prompt, max_new_tokens, model_dir="./Llama-3.2-1B_instruct_openvino"):
    
    pipe = ov_genai.LLMPipeline(model_dir, "CPU")
    return pipe.generate(prompt, max_new_tokens=max_new_tokens) 


prompt = """\n\n
Your task is to create a 30 second engaging and educational tiktok script based on the following sentence:

{input_sentence}

Expand on this sentence to create an interesting and educational script that most people might not know about.
The tiktok should incorporate an engaging story or example related to the sentence.
Do not include any emojis or hashtags in the script.
The script should be only spoken text, no extra text like [Cut] or [Music].
The script should sound passionate, excited, and happy.

Script:
"""
user_input = prompt.format(input_sentence="Spaceships are the future of human travel.")
script = generate_text(user_input, 1000)
script