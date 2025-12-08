import os
from utils.llm import *
from utils.vlm import *
from utils.gemini_llm import *



gemini = gemini_model()


def extract_generated_text_vlm(raw_output: str):
    """VLM output includes input as well, this slices out only generated tokens."""
    raw_output = raw_output.strip()

    if "assistant" in raw_output:
        idx = raw_output.index("assistant") + len("assistant")
        return raw_output[idx:].strip()

    return raw_output


# CONFIG
VIDEO_DIR = "pipeline/copa_videos"
TRANSCRIPT_DIR = "pipeline/whisper_transcripts"
OUTPUT_DIR = "pipeline/output_results_whisper"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_pair(video_path, transcript_text, index):
    print(f"\nProcessing Video {index}...")

    non_qa_caption = gemini.non_QA_prompt(transcript_text, video_path)

    output_file = os.path.join(OUTPUT_DIR, f"Video{index}_results.txt")
    with open(output_file, "w") as f:
        f.write(f"VIDEO: {video_path}\n\n")
        f.write(f"=== NON-QA CAPTION ===\n{non_qa_caption.text}\n\n")

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
