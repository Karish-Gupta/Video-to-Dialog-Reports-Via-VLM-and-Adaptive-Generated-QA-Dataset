from llm import *
from vlm import *
from transcript_context import *

def extract_generated_text_vlm(raw_output: str):
    """VLM ouput inlcude input as well, this can be used to keep just generated tokens"""
    raw_output = raw_output.strip()

    if "assistant" in raw_output:
        idx = raw_output.index("assistant") + len("assistant")
        return raw_output[idx:].strip()

    # If there's no "assistant:", return full output
    return raw_output


# Initialize LLM
llm_model = "meta-llama/Llama-3.3-70B-Instruct"
llm_ = llm(llm_model)

vlm_model_name = "llava-hf/LLaVA-NeXT-Video-34B-hf"
vlm_ = vlm(vlm_model_name)

video_2_path = "VLM/videos/highway_bodycam_one_min.mp4"


# VLM summary
vlm_conversation = vlm_.build_conversation()
vlm_summary = vlm_.invoke(video_2_path, vlm_conversation)
vlm_summary = extract_generated_text_vlm(vlm_summary)
print(f"VLM Summary:\n{vlm_summary}")

# Step 1 prompt
step_1_prompt = llm_.step_1_chat_template(transcript_60_sec, vlm_summary)
# print(f"Step 1 Prompt:\n {step_1_prompt}")

structured_output = llm_.invoke(step_1_prompt)
print(f"Generated Structured Elements:\n {structured_output}")


# Step 2 prompt
step_2_prompt = llm_.step_2_chat_template(structured_output)
# print(f"Step 2 Prompt:\n {step_2_prompt}")

generated_qs = llm_.invoke(step_2_prompt)
print(f"Generated Questions:\n {generated_qs}")

# Pass generated questions to VLM for answer generation
qa_conversation = vlm_.build_qa_conversation(generated_qs)
print (f"QA Prompt:\n {qa_conversation}")

vlm_answers = vlm_.invoke(video_2_path, qa_conversation)
vlm_answers = extract_generated_text_vlm(vlm_answers)
print(f"VLM Generated Answers:\n {vlm_answers}") 

# Generate video captions
qa_caption_prompt = llm_.qa_caption_chat_template(generated_qs, vlm_answers, transcript_60_sec, vlm_summary)
qa_caption = llm_.invoke(qa_caption_prompt)
print(f"Generated QA Caption:\n {qa_caption}")

non_qa_caption_prompt = llm_.caption_chat_template(transcript_60_sec, vlm_summary)
non_qa_caption = llm_.invoke(non_qa_caption_prompt)
print(f"Generated NO QA Caption:\n {non_qa_caption}")
