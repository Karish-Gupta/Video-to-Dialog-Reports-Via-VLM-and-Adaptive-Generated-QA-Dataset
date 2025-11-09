from llm import *
from vlm import *
from transcript_context import *

# Initialize LLM
llm_model = "meta-llama/Llama-3.3-70B-Instruct"
llm_ = llm(llm_model)

vlm_model_name = "llava-hf/LLaVA-NeXT-Video-34B-hf"
vlm_ = vlm(vlm_model_name)

# Step 1 prompt
step_1_prompt = llm_.step_1_chat_template(transcript_60_sec)
print(f"Step 1 Prompt: {step_1_prompt}")

structured_output = llm_.invoke(step_1_prompt)
print(f"Generated Structured Elements: {structured_output}")


# Step 2 prompt
step_2_prompt = llm_.step_2_chat_template(structured_output)
print(f"Step 2 Prompt: {step_2_prompt}")

generated_qs = llm_.invoke(step_2_prompt)
print(f"Generated Questions: {generated_qs}")

# Pass generated questions to VLM for answer generation
video_1_path = "VLM/videos/high_way_bodycam_30_sec.mp4"

qa_conversation = vlm_.build_qa_conversation(generated_qs)
print (f"QA Prompt: {qa_conversation}")

vlm_answers = vlm_.invoke(video_1_path, qa_conversation)
print(f"VLM Generated Answers: {vlm_answers}") 
