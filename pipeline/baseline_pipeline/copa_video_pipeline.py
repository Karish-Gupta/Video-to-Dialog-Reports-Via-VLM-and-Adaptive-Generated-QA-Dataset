import os
from models.llm import *
from models.gemini_model import *
import re

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



print("\n--- Indexing Files ---")

# Helper function to extract ID
def extract_id(filename):
    # Looks for a sequence of digits in the filename
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

# Build maps: { 'id': 'filename' }
video_map = {}
for f in os.listdir(VIDEO_DIR):
    if f.lower().endswith(".mp4"):
        vid_id = extract_id(f)
        if vid_id:
            video_map[vid_id] = f

transcript_map = {}
for f in os.listdir(TRANSCRIPT_DIR):
    if f.lower().endswith(".txt"):
        # Handles "Video191Transcript..." -> extracts "191"
        trans_id = extract_id(f)
        if trans_id:
            transcript_map[trans_id] = f

# Find common IDs (files present in BOTH folders)
common_ids = set(video_map.keys()) & set(transcript_map.keys())

# Sort numerically (so 2 comes before 10)
sorted_ids = sorted(list(common_ids), key=int)

print(f"Found {len(sorted_ids)} valid pairs.")
print(f"Skipped {len(video_map) - len(sorted_ids)} unmatched videos.")

print("\n Starting processing pipeline...\n")

for index in sorted_ids:
    video_file = video_map[index]
    transcript_file = transcript_map[index]

    video_path = os.path.join(VIDEO_DIR, video_file)
    transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_file)

    try:
        with open(transcript_path, "r") as t:
            transcript_text = t.read()

        # Pass the extracted index and text to your processing function
        process_pair(video_path, transcript_text, index)
    
    except Exception as e:
        print(f"ERROR processing Video {index}: {e}")

print("\n All videos processed successfully!")