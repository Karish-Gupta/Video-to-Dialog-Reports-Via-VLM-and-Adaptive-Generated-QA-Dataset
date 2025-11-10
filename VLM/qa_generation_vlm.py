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

# VLM summary 
vlm_conversation = vlm_.build_conversation(transcript_60_sec)
vlm_summary = vlm_.invoke(video_2_path, vlm_conversation)
print(f"VLM Summary:\n{vlm_summary}")

# LLM 2 step data question generation

# Step 1 prompt
step_1_prompt = llm_.step_1_chat_template(vlm_summary)
print(f"Step 1 Prompt: {step_1_prompt}")

structured_output = llm_.invoke(step_1_prompt)
print(f"Generated Structured Elements: {structured_output}")


# Step 2 prompt
step_2_prompt = llm_.step_2_chat_template(structured_output)
print(f"Step 2 Prompt: {step_2_prompt}")

generated_qs = llm_.invoke(step_2_prompt)
print(f"Generated Questions: {generated_qs}")
