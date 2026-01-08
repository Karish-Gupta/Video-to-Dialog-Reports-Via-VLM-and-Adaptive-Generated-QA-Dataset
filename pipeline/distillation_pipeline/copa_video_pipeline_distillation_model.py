import os
from models.llm import *
from models.gemini_model import *
from models.distillation_ft_llm import *


# CONFIG
VIDEO_DIR = "pipeline/copa_videos"
TRANSCRIPT_DIR = "pipeline/whisper_transcripts_diarize"
OUTPUT_DIR = "pipeline/distillation_captions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Init (done once)
vlm_model_name = "gemini-2.5-flash"
gemini = gemini_model(vlm_model_name)
question_generation_model = distillation_ft_llm()

def process_pair(video_path, transcript_text, index):
    print(f"\nProcessing Video {index}...")

    # Step 1: VLM Summary
    print("\n Generating VLM Summary...")
    vlm_summary = gemini.generate_vlm_summary(video_path, transcript_text)

    # Step 2: LLM Extraction
    print("\n Extracting structured output...")
    structured_output = gemini.generate_structured_details(vlm_summary)

    # Step 3: Generate Questions with question generation model
    print("\n Generating questions...")
    step_2_prompt = question_generation_model.step_2_chat_template(structured_output)
    generated_qs = question_generation_model.invoke(step_2_prompt)

    # Step 4: Ask VLM to Answer
    print("\n Getting VLM answers to generated questions...")
    vlm_answers = gemini.answer_questions(video_path, generated_qs)

    # Generate Captions
    print("→ Creating QA captions...")
    qa_caption = gemini.generate_qa_caption(vlm_summary, vlm_answers)

    # ---- Save Results ----
    output_file = os.path.join(OUTPUT_DIR, f"Video{index}_results.txt")
    with open(output_file, "w") as f:
        f.write(f"VIDEO: {video_path}\n\n")
        f.write(f"=== VLM SUMMARY ===\n{vlm_summary}\n\n")
        f.write(f"=== STRUCTURED OUTPUT ===\n{structured_output}\n\n")
        f.write(f"=== GENERATED QUESTIONS ===\n{generated_qs}\n\n")
        f.write(f"=== VLM ANSWERS ===\n{vlm_answers}\n\n")
        f.write(f"=== QA CAPTION ===\n{qa_caption}\n\n")

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
