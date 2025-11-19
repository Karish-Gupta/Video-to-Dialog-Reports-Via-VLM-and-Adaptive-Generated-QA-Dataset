from llm import *
from ground_truths import copa_video_ground_truths
import os 

OUTPUT_DIR = "VLM/output_results_whisper" # Folder with each video caption output

# Initialize model
llm_model = "meta-llama/Llama-3.3-70B-Instruct"
llm_ = llm(llm_model)

