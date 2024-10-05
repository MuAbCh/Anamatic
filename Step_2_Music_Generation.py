# Step 2
from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

music = synthesiser("A high-energy, futuristic electronic dance track with a driving beat and synthesized leads, perfect for a space-themed video that\'s out of this world!", forward_params={"do_sample": True})

scipy.io.wavfile.write("output/musicgen_out.mp3", rate=music["sampling_rate"], data=music["audio"])