import os
import whisperx

hf_token = os.getenv("HF_TOKEN")
   
VIDEO_DIR = "VLM/copa_videos"
OUTPUT_DIR = "VLM/whisper_transcripts"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load WhisperX model
device = "cuda"
model = whisperx.load_model("large-v2", device)

# Sort videos to keep correct order (Video1, Video2, ...)
videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().startswith("video")])

for video in videos:
   video_path = os.path.join(VIDEO_DIR, video)

   # Extract number from file name (expects format "VideoX.mp4")
   video_name = video.split(".")[0] # Remove file extension 
   video_number = ''.join(filter(str.isdigit, video_name))
   transcript_filename = f"Transcript{video_number}.txt"
   transcript_path = os.path.join(OUTPUT_DIR, transcript_filename)

   print(f"Processing: {video} -> {transcript_filename}")

   # Transcribe audio
   audio = whisperx.load_audio(video_path)
   result = model.transcribe(audio)

# Extract text (no alignment or diarization)
   transcript_text = "\n".join([segment["text"].strip() for segment in result["segments"]])

   # Save transcript
   with open(transcript_path, "w", encoding="utf-8") as f:
      f.write(transcript_text)

   print(f"Saved: {transcript_path}")

print("\n All transcripts created")