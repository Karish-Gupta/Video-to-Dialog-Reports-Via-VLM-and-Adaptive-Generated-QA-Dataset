from models.gemini_model import gemini_model
from pathlib import Path
import json
import re

def extract_number(path):
    """
    Sorting function with regex to ensure proper indexing
    """
    match = re.search(r'(\d+)', path.stem)
    return int(match.group(1)) if match else float('inf')

def parse_gemini_output(gt_str):
    """
    Cleans Gemini output by removing markdown fences and parses JSON.
    """   
    # Remove ```json or ``` at start and end
    gt_str = gt_str.strip()
    gt_str = re.sub(r"^```json\s*", "", gt_str)
    gt_str = re.sub(r"^```\s*", "", gt_str)
    gt_str = re.sub(r"\s*```$", "", gt_str)
    
    # In case of new lines after removal
    gt_str = gt_str.strip()
    
    try:
        return json.loads(gt_str)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        # Fail state empty structure
        return {
            "important_details": [],
            "visual_enrichment_details": [],
            "auxiliary_details": [],
            "transcript": ""
        }


def generate_gts(transcript_dir, video_dir):
    transcripts_path_list = sorted(Path(transcript_dir).glob("*.txt"), key=extract_number)
    videos_path_list = sorted(Path(video_dir).glob("*.mp4"), key=extract_number)

    gemini = gemini_model()
    output = {}

    for t_path, v_path in zip(transcripts_path_list, videos_path_list):

        # Extract number from transcript filename
        num = extract_number(t_path)

        with open(t_path, "r", encoding="utf-8") as file:
            transcript_txt = file.read()
            
        ground_truth_str = gemini.generate_ground_truths(transcript_txt, v_path)
        print(ground_truth_str)
        
        ground_truth = parse_gemini_output(ground_truth_str)
        output[f"video{num}"] = ground_truth
        
    with open("generated_gts.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        

if __name__ == '__main__':
    transcript_dir = "data/transcripts"
    video_dir = "data/videos"
    
    generate_gts(transcript_dir, video_dir)
    