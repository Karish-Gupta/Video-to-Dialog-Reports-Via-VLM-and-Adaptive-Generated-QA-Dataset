import os
from models.llm import *
from models.vlm import *
from models.gemini_model import *
import csv

# CONFIG
VIDEO_DIR = "VLM/training_videos"
TRANSCRIPT_DIR = "VLM/training_transcripts"
OUTPUT_DIR = "VLM/distillation_training_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "distillation_results.csv")

# Create CSV with headers if not already present
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["video_index", "vlm_summary", "questions"])  # headers

# Model Init
llm_model = "meta-llama/Llama-3.3-70B-Instruct"
vlm_model_name = "llava-hf/LLaVA-NeXT-Video-34B-hf"
llm_ = llm(llm_model)
vlm_ = vlm(vlm_model_name)
gemini = gemini_model()


def extract_generated_text_vlm(raw_output: str):
    """VLM output includes input as well, this slices out only generated tokens."""
    raw_output = raw_output.strip()

    if "assistant" in raw_output:
        idx = raw_output.index("assistant") + len("assistant")
        return raw_output[idx:].strip()

    return raw_output

def process_pair(video_path, transcript_text, index):
    print(f"\nProcessing Video {index}...")

    # Step 1: VLM Summary
    vlm_conversation = vlm_.build_conversation()
    vlm_summary = vlm_.invoke(video_path, vlm_conversation)
    vlm_summary = extract_generated_text_vlm(vlm_summary)

    # Step 2: LLM Extraction
    step_1_prompt = llm_.step_1_chat_template(transcript_text, vlm_summary)
    structured_output = llm_.invoke(step_1_prompt)
    
    # Step 3: Generate questions
    questions = gemini.generate_distillation_model_qs(structured_output)

    # ---- Append to CSV ----
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([index, vlm_summary, structured_output, questions])

    print(f"Finished Video {index} appended to CSV")

def main():
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

    print("\n All videos processed successfully! CSV Created at:", CSV_PATH)
    
    
if __name__ == "__main__":
    main()