import os
from models.llm import *
from models.gemini_model import *

# CONFIG
VIDEO_DIR = "data/eval_videos"
TRANSCRIPT_DIR = "data/eval_transcripts"
OUTPUT_DIR = "pipeline/baseline_captions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Init (done once)
vlm_model_name = "gemini-2.5-flash"
gemini = gemini_model(vlm_model_name)


def process_pair(video_path, transcript_text, index):
    print(f"\nProcessing Video {index}...")

    # Step 1: VLM Summary with Gemini Model
    print("\n Generating VLM Summary...")
    vlm_summary = gemini.generate_vlm_summary(video_path, transcript_text)

    # Step 2: LLM Extraction
    print("\n Extracting structured output...")
    structured_output = gemini.generate_structured_details(vlm_summary)

    # Step 3: Generate Questions
    print("\n Generating questions...")
    generated_qs = gemini.generate_questions(structured_output)

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
