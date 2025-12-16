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
llm_model = "meta-llama/Llama-3.3-70B-Instruct"
llm_ = llm(llm_model)
vlm_ = gemini_model()

def process_pair(video_path, transcript_text, index):
    print(f"\nProcessing Video {index}...")

    # Step 1: VLM Summary Gemini Model
    print("\n Generating VLM Summary...")
    prompt = f"""
        This is a police bodycam video. Describe what happens in this video in detail, focus on actions, reponses, details about people and the surroundings. Be specific.
        """
    vlm_summary = vlm_.vlm_invoke(video_path, prompt)

    # Step 2: LLM Extraction
    step_1_prompt = llm_.step_1_chat_template(transcript_text, vlm_summary)
    structured_output = llm_.invoke(step_1_prompt)

    # Step 3: Generate questions
    question_generation_prompt = f"""
    You are an AI assistant aiding law enforcement analysts reviewing body-worn camera footage.

    Your task:
    - Based on the provided structured details, generate a list of investigative questions.
    - Every question must be something a human could answer by watching the video.
    - The goal is to guide analysts toward visual clues, context, behavior, or environment details that may matter.

    Rules for your output:
    - Write 1 meaningful question per detail element.
    - Do NOT repeat facts already stated — ask what is *unknown or unclear* visually.
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