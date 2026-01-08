import os
from models.llm import *
from models.vlm import *
from models.gemini_model import *

# CONFIG
VIDEO_DIR = "pipeline/copa_videos"
TRANSCRIPT_DIR = "pipeline/whisper_transcripts_diarize"
OUTPUT_DIR = "pipeline/gemini_vlm_captions"

os.makedirs(OUTPUT_DIR, exist_ok=True)
gemini = gemini_model()


def process_pair(video_path, transcript_text, index):
    print(f"\nProcessing Video {index}...")

    prompt = f"""
        You are given a bodycam video transcript, and the video.
        Generate a caption that gives visual details about the video. 
        Include the following in caption: 

        - Describe the setting (Time of day, vehicles, buildings, etc.)
        - Objects in the frame (Weapons, items in hand, consumables, etc.)
        - Describe how items are being used (Is a weapon being fired, radio being held by officer, etc.)
        - Describe individuals (What are people wearing, color of vehicles, accessory items worn such as hats or glasses, etc.)
        - Actions each individual made (Officer stating instructions, civilians complying, etc.)

        Ensure captions are direct and formal.

        Write in active voice as much as possible.
        Be as detailed as possible.
        Use direct quotes only when needed.
        Use a person's name if it is known.

        Transcipt:
        {transcript_text}
      """

    non_qa_caption = gemini.vlm_invoke(video_path, prompt)

    output_file = os.path.join(OUTPUT_DIR, f"Video{index}_results.txt")
    with open(output_file, "w") as f:
        f.write(f"VIDEO: {video_path}\n\n")
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
