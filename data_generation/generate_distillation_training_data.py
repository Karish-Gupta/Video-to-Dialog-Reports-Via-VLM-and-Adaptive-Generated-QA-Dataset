import os
import json
from models.llm import *
from models.vlm import *
from models.gemini_model import *

# CONFIG
VIDEO_DIR = "data_generation/training_videos"
TRANSCRIPT_DIR = "data_generation/training_transcripts"
OUTPUT_DIR = "data_generation/distillation_training_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

JSONL_PATH = os.path.join(OUTPUT_DIR, "distillation_results_gemini.jsonl")

# Model Init
#llm_model = "meta-llama/Llama-3.3-70B-Instruct"
#llm_ = llm(llm_model)
vlm_ = gemini_model()

def process_pair(video_path, transcript_text, index):
    print(f"\nProcessing Video {index}...")

    # Step 1: VLM Summary Gemini Model
    print("\n Generating VLM Summary...")

    vlm_summary = vlm_.generate_vlm_summary(video_path, transcript_text)

    # Step 2: LLM Extraction
    structured_output = vlm_.generate_structured_details(vlm_summary)

    # Step 3: Generate questions
    question_generation_prompt = f"""
    You are an AI assistant aiding law enforcement analysts reviewing body-worn camera footage.

    Your task:
    - Based on the provided structured details, generate a list of investigative questions.
    - Every question must be something a human could answer by watching the video.
    - The goal is to guide analysts toward visual clues, context, behavior, or environment details that may matter.

    Rules for your output:
    - Write a total of 4 meaningful questions that can extract the most visual information.
    - Each question should pertain to one of the four categories (scene-level, entity-level, action-level, semantic-level).
    - Do NOT repeat facts already stated.
    - Focus areas include: body language, environment, timeline, objects, threat indicators, interaction dynamics, or visual anomalies.
    - Use clear, concise, professional language.
    - Format the output as a numbered list.

    Structured information provided:
    {structured_output}
    """
    questions = vlm_.invoke(question_generation_prompt)

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
    print(f"\nTotal video-transcript pairs to process: {len(video_files)}")
    for video_file, transcript_file in pairs:
        index = ''.join(filter(str.isdigit, os.path.splitext(video_file)[0]))
        video_path = os.path.join(VIDEO_DIR, video_file)
        transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_file)
        print(f"Processing pair: {video_path} & {transcript_path}")
        with open(transcript_path, "r") as t:
            transcript_text = t.read()

        process_pair(video_path, transcript_text, index)
        break # Remove this break to process all videos

    print("\n All videos processed successfully! JSONL saved at:", JSONL_PATH)


if __name__ == "__main__":
    main()