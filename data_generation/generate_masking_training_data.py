import os
import json
from dotenv import load_dotenv
from models.llm import *
from models.gemini_model import *

# CONFIG
VIDEO_DIR = "data_generation/training_videos"
TRANSCRIPT_DIR = "data_generation/training_transcripts"
OUTPUT_DIR = "data_generation/masking_training_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

JSONL_PATH = os.path.join(OUTPUT_DIR, "masking_results.jsonl")

# Load API key from env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("No GEMINI_API_KEY found in .env file")

# Model Init
llm_model = "meta-llama/Llama-3.3-70B-Instruct"
gemini_vlm_model = "gemini-2.5-flash-lite"

llm_ = llm(llm_model)
gemini = gemini_model(model_name=gemini_vlm_model)


def process_pair(video_path, transcript_text, index):
    print(f"\nProcessing Video {index}...")

    # Step 1: Gemini VLM Summary
    vlm_summary = gemini.non_QA_prompt(transcript_text, video_path)
    vlm_summary = vlm_summary.text  # Extract text from Gemini response

    # Step 2: LLM Extraction
    step_1_prompt = llm_.step_1_chat_template(transcript_text, vlm_summary)
    structured_output = llm_.invoke(step_1_prompt)

    # Step 3: Generate questions
    questions = gemini.generate_distillation_model_qs(structured_output)
    questions = questions.text  # Extract text from Gemini response

    # ---- Append to JSONL ----
    record = {
        "video_index": index,
        "vlm_summary": vlm_summary,
        "structured_details": structured_output,
        "questions": questions
    }

    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Finished Video {index} appended to JSONL file")

def main():
    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().startswith("video")])
    transcript_files = sorted([f for f in os.listdir(TRANSCRIPT_DIR) if f.lower().startswith("transcript")])

    pairs = zip(video_files, transcript_files)

    print("\n Starting processing pipeline...\n")

    for video_file, transcript_file in pairs:
        index = ''.join(filter(str.isdigit, os.path.splitext(video_file)[0]))
        video_path = os.path.join(VIDEO_DIR, video_file)
        transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_file)

        with open(transcript_path, "r") as t:
            transcript_text = t.read()

        process_pair(video_path, transcript_text, index)

    print("\n All videos processed successfully! JSONL saved at:", JSONL_PATH)


if __name__ == "__main__":
    main()