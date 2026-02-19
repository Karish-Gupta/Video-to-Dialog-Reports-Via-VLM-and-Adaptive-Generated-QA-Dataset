from models.gemini_model import GeminiModel
from pathlib import Path
import json
import re
import time
import os

# Configuration
BATCH_SIZE = 5         # Save file every 5 videos
SLEEP_BETWEEN_CALLS = 4 # Seconds to wait between API calls
OUTPUT_FILE = "data_generation/ground_truth_generation/generated_gts.json"

def extract_number(path):
    match = re.search(r'(\d+)', path.stem)
    return int(match.group(1)) if match else float('inf')

def parse_gemini_output(gt_str):
    gt_str = gt_str.strip()
    gt_str = re.sub(r"^```json\s*", "", gt_str)
    gt_str = re.sub(r"^```\s*", "", gt_str)
    gt_str = re.sub(r"\s*```$", "", gt_str)
    
    try:
        return json.loads(gt_str)
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        return None # Return None so we can identify failed parses

def load_existing_data(filepath):
    """Resumes progress if file exists."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def generate_gts(transcript_dir, video_dir):
    # Setup paths
    transcripts_path_list = sorted(Path(transcript_dir).glob("*.txt"), key=extract_number)
    videos_path_list = sorted(Path(video_dir).glob("*.mp4"), key=extract_number)
    
    # Initialize Model
    gemini = GeminiModel()
    
    # Load existing progress
    output = load_existing_data(OUTPUT_FILE)
    print(f"Loaded {len(output)} existing records.")

    # Filter out already processed videos
    # Pair them first, then check if the key exists in output
    pairs_to_process = []
    for t_path, v_path in zip(transcripts_path_list, videos_path_list):
        num = extract_number(t_path)
        key = f"video{num}"
        if key not in output:
            pairs_to_process.append((key, t_path, v_path))
    
    print(f"Remaining videos to process: {len(pairs_to_process)}")

    # Process in Batches
    total_processed_in_run = 0
    
    for i, (key, t_path, v_path) in enumerate(pairs_to_process):
        print(f"--- Processing {key} ({i+1}/{len(pairs_to_process)}) ---")
        
        try:
            with open(t_path, "r", encoding="utf-8") as file:
                transcript_txt = file.read()
            
            # Call Model (This is where the upload/wait happens)
            ground_truth_str = gemini.generate_ground_truths(transcript_txt, v_path)
            
            # Parse
            ground_truth = parse_gemini_output(ground_truth_str)
            
            if ground_truth:
                output[key] = ground_truth
                print(f"Successfully processed {key}")
            else:
                print(f"Skipping {key} due to parse error.")
                output[key] = {"error": "JSON Parse Failure"} 

            total_processed_in_run += 1

            # Rate Limit Sleep
            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            print(f"CRITICAL ERROR on {key}: {e}")
            with open("processing_errors.log", "a") as err_f:
                err_f.write(f"{key}: {str(e)}\n")

        # Save Batch
        if total_processed_in_run % BATCH_SIZE == 0:
            print(f">>> Saving checkpoint at {key}...")
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

    # Final Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("Processing complete.")

if __name__ == '__main__':
    transcript_dir = "data/transcripts"
    video_dir = "data/videos"
    
    generate_gts(transcript_dir, video_dir)