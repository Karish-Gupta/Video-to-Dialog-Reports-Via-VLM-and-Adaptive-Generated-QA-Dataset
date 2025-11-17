import os
import whisperx

hf_token = os.environ.get("HF_TOKEN", None) # Huggingface token

VIDEO_DIR = "VLM/copa_videos"
OUTPUT_DIR = "VLM/whisper_transcripts"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load WhisperX model
device = "cuda"
model = whisperx.load_model("large-v2", device)
diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

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

   # Apply speaker diarization
   diarization = diarize_model(audio)

   # Assign speakers to segments 
   # This merges Whisper segments with diarization timestamps
   result_with_spk = whisperx.assign_word_speakers(diarization, result["segments"])

   # Format transcript
   transcript_lines = []
   for seg in result_with_spk:
      speaker = seg.get("speaker", "Unknown Speaker")
      text = seg["text"].strip()
      transcript_lines.append(f"{speaker}: {text}")

   transcript_text = "\n".join(transcript_lines)

   # Save transcript
   with open(transcript_path, "w", encoding="utf-8") as f:
      f.write(transcript_text)

   print(f"Saved: {transcript_path}")

print("\n All transcripts with diarization created")