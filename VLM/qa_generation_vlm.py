from vlm import vlm
from llm import llm
from transcript_context import *

vlm_model_name = "llava-hf/LLaVA-NeXT-Video-34B-hf"
llm_model_name = "meta-llama/Meta-Llama-3-70B"

# Load & sample video frames
video_1_path = "VLM/videos/high_way_bodycam_30_sec.mp4"
video_2_path = "VLM/videos/highway_bodycam_one_min.mp4"
video_3_path = "VLM/videos/highway_bodycam.mp4"

vlm_ = vlm(vlm_model_name)
llm_ = llm(llm_model_name)
