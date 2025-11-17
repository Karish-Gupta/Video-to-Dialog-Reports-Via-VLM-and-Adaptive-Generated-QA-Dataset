import os
from CoT import *


def extract_generated_text_vlm(raw_output: str):
    """VLM output includes input as well, this slices out only generated tokens."""
    raw_output = raw_output.strip()

    if "assistant" in raw_output:
        idx = raw_output.index("assistant") + len("assistant")
        return raw_output[idx:].strip()

    return raw_output


# CONFIG
VIDEO_DIR = "VLM/VLM_Prompting/copa_videos"
TRANSCRIPT_DIR = "VLM/VLM_Prompting/copa_transcripts"
OUTPUT_DIR = "VLM/VLM_Prompting/zero_shot_output_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Init (done once)
vlm_model_name = "llava-hf/LLaVA-NeXT-Video-34B-hf"
vlm_ = vlm(vlm_model_name)


def process_pair(video_path, transcript_text, index):
    print(f"\nProcessing Video {index}...")

    # Step 1: VLM Summary
    print("\n Generating VLM Summary...")
    vlm_conversation = vlm_.zero_shot_prompting()
    vlm_summary = vlm_.invoke(video_path, vlm_conversation)
    vlm_summary = extract_generated_text_vlm(vlm_summary)
    print(f"VLM Summary:\n{vlm_summary}")

    # ---- Save Results ----
    output_file = os.path.join(OUTPUT_DIR, f"Video{index}_results.txt")
    with open(output_file, "w") as f:
        f.write(f"VIDEO: {video_path}\n\n")
        f.write(f"=== VLM SUMMARY ===\n{vlm_summary}\n\n")

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
