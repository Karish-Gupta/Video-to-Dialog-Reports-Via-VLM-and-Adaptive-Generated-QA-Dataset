import os
from llm import *
from vlm import *


def extract_generated_text_vlm(raw_output: str):
    """VLM output includes input as well, this slices out only generated tokens."""
    raw_output = raw_output.strip()

    if "assistant" in raw_output:
        idx = raw_output.index("assistant") + len("assistant")
        return raw_output[idx:].strip()

    return raw_output


# CONFIG
VIDEO_DIR = "VLM/copa_videos"
TRANSCRIPT_DIR = "VLM/whisper_transcripts_diarize"
OUTPUT_DIR = "VLM/output_results_whisper_diarize"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Init (done once)
llm_model = "meta-llama/Llama-3.3-70B-Instruct"
vlm_model_name = "llava-hf/LLaVA-NeXT-Video-34B-hf"
llm_ = llm(llm_model)
vlm_ = vlm(vlm_model_name)


def process_pair(video_path, transcript_text, index):
    print(f"\nProcessing Video {index}...")

    # Step 1: VLM Summary
    print("\n Generating VLM Summary...")
    vlm_conversation = vlm_.build_conversation()
    vlm_summary = vlm_.invoke(video_path, vlm_conversation)
    vlm_summary = extract_generated_text_vlm(vlm_summary)

    # Step 2: LLM Extraction
    print("\n Extracting structured output...")
    step_1_prompt = llm_.step_1_chat_template(transcript_text, vlm_summary)
    structured_output = llm_.invoke(step_1_prompt)

    # Step 3: Generate Questions
    print("\n Generating questions...")
    step_2_prompt = llm_.step_2_chat_template(structured_output)
    generated_qs = llm_.invoke(step_2_prompt)

    # Step 4: Ask VLM to Answer
    print("\n Getting VLM answers to generated questions...")
    qa_conversation = vlm_.build_qa_conversation(generated_qs)
    vlm_answers = vlm_.invoke(video_path, qa_conversation)
    vlm_answers = extract_generated_text_vlm(vlm_answers)

    # Generate Captions
    print("→ Creating QA captions...")
    qa_caption_prompt = llm_.qa_caption_chat_template(generated_qs, vlm_answers, transcript_text, vlm_summary)
    qa_caption = llm_.invoke(qa_caption_prompt)

    print("→ Creating NON-QA captions...")
    non_qa_caption_prompt = llm_.caption_chat_template(transcript_text, vlm_summary)
    non_qa_caption = llm_.invoke(non_qa_caption_prompt)

    # ---- Save Results ----
    output_file = os.path.join(OUTPUT_DIR, f"Video{index}_results.txt")
    with open(output_file, "w") as f:
        f.write(f"VIDEO: {video_path}\n\n")
        f.write(f"=== VLM SUMMARY ===\n{vlm_summary}\n\n")
        f.write(f"=== STRUCTURED OUTPUT ===\n{structured_output}\n\n")
        f.write(f"=== GENERATED QUESTIONS ===\n{generated_qs}\n\n")
        f.write(f"=== VLM ANSWERS ===\n{vlm_answers}\n\n")
        f.write(f"=== QA CAPTION ===\n{qa_caption}\n\n")
        f.write(f"=== NON-QA CAPTION ===\n{non_qa_caption}\n\n")

    print(f"Finished Video {index} saved to {output_file}")



# Identify all video-to-transcript pairs
video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().startswith("video")])
transcript_files = sorted([f for f in os.listdir(TRANSCRIPT_DIR) if f.lower().startswith("transcript")])

pairs = zip(video_files, transcript_files)

print("\n Starting processing pipeline...\n")

for video_file, transcript_file in pairs:
    # Remove extension to extract correct index
    video_name_without_ext = os.path.splitext(video_file)[0]
    index = ''.join(filter(str.isdigit, video_name_without_ext))

    # Use full filename (with extension) for actual file path
    video_path = os.path.join(VIDEO_DIR, video_file)
    transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_file)

    with open(transcript_path, "r") as t:
        transcript_text = t.read()

    process_pair(video_path, transcript_text, index)

print("\n All videos processed successfully!")
