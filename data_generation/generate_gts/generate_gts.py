from models.gemini_model import gemini_model
from pathlib import Path
import json

transcript_dir = "data/transcripts"
video_dir = "data/videos"

transcripts_path_list = sorted(Path(transcript_dir).glob("*.txt"))
videos_path_list = sorted(Path(video_dir).glob("*.mp4"))


gemini = gemini_model()

output = {}

for index, (t_path, v_path) in enumerate(zip(transcripts_path_list, videos_path_list)):
    
    with open(t_path, "r", encoding="utf-8") as file:
        transcript_txt = file.read()
        
    ground_truth = gemini.generate_ground_truths(transcript_txt, v_path)
    output[f"video{index + 1}"] = ground_truth
    
with open("generated_gts.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
    