import whisperx
import yt_dlp
import os
import json
from datetime import datetime
import torch

def download_audio(youtube_url, output_path="audio"):
    """Download audio from YouTube video"""
    print(f"Downloading audio from: {youtube_url}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'quiet': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        # Get the actual downloaded filename
        audio_file = ydl.prepare_filename(info)
        return audio_file, info

def transcribe_audio_with_diarization(audio_file, model_size="base", device="cuda", compute_type="float16", hf_token=None):
    """Transcribe audio using WhisperX with speaker diarization"""
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        compute_type = "int8"
    
    print(f"\n{'='*80}")
    print(f"Step 1: Loading WhisperX model ({model_size}) on {device}")
    print(f"{'='*80}")
    
    # Load WhisperX model
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    
    # Load audio
    print(f"\nLoading audio file: {audio_file}")
    audio = whisperx.load_audio(audio_file)
    
    # Transcribe with WhisperX
    print(f"\n{'='*80}")
    print(f"Step 2: Transcribing audio")
    print(f"{'='*80}")
    result = model.transcribe(audio, batch_size=16)
    
    # Align whisper output
    print(f"\n{'='*80}")
    print(f"Step 3: Aligning transcription")
    print(f"{'='*80}")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Perform speaker diarization (requires HF token)
    if hf_token:
        print(f"\n{'='*80}")
        print(f"Step 4: Performing speaker diarization")
        print(f"{'='*80}")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print("Speaker diarization completed!")
    else:
        print(f"\n{'='*80}")
        print("Note: Skipping speaker diarization (no HuggingFace token provided)")
        print("To enable diarization, set HF_TOKEN environment variable or pass hf_token parameter")
        print(f"{'='*80}")
    
    return result

def save_transcript(result, output_file="transcript.txt", json_file="transcript.json", timestamped_file="transcript_timestamped.txt"):
    """Save transcript to text and JSON files"""
    
    # Build full transcript text
    full_text = ""
    current_speaker = None
    
    if "segments" in result:
        for segment in result["segments"]:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "")
            
            # Add speaker label if it changed
            if speaker != current_speaker:
                if full_text:
                    full_text += "\n\n"
                full_text += f"[{speaker}]: "
                current_speaker = speaker
            
            full_text += text
    else:
        full_text = result.get("text", "")
    
    # Save plain text transcript
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    print(f"\nTranscript saved to: {output_file}")
    
    # Save detailed JSON with timestamps
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Detailed transcript with timestamps saved to: {json_file}")
    
    # Save human-readable timestamped transcript
    with open(timestamped_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TRANSCRIPT WITH TIMESTAMPS\n")
        f.write("="*80 + "\n\n")
        
        if "segments" in result:
            for segment in result["segments"]:
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                speaker = segment.get("speaker", "Unknown")
                text = segment.get("text", "").strip()
                
                # Format timestamps as HH:MM:SS.mmm
                start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{start%60:06.3f}"
                end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{end%60:06.3f}"
                
                f.write(f"[{start_time} --> {end_time}] [{speaker}]\n")
                f.write(f"{text}\n\n")
    
    print(f"Human-readable timestamped transcript saved to: {timestamped_file}")
    
    # Print the transcript
    print("\n" + "="*80)
    print("TRANSCRIPT:")
    print("="*80)
    print(full_text)
    print("="*80)


def main():
    # YouTube video URL (HARDCODED AT THE MOMENT)
    youtube_url = "https://www.youtube.com/watch?v=83jt-xOJok4"
    
    # Create output dir
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download audio
    audio_file, video_info = download_audio(youtube_url, output_path=output_dir)
    print(f"\nVideo title: {video_info['title']}")
    print(f"Duration: {video_info['duration']} seconds")
    
    # Transcribe using WhisperX w/ diarization
    # Model sizes: tiny, base, small, medium, large-v2, large-v3
    model_size = "base"
    
    # Set device and compute type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    # Get HuggingFace token from env
    hf_token = os.environ.get("HF_TOKEN", None)
    
    print(f"\nUsing device: {device}")
    print(f"Compute type: {compute_type}")
    if hf_token:
        print("HuggingFace token found - speaker diarization will be enabled")
    else:
        print("No HuggingFace token - speaker diarization will be skipped")
    
    result = transcribe_audio_with_diarization(
        audio_file, 
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        hf_token=hf_token
    )
    
    # Save transcript
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_output = f"{output_dir}/transcript_{timestamp}.txt"
    json_output = f"{output_dir}/transcript_{timestamp}.json"
    timestamped_output = f"{output_dir}/transcript_{timestamp}_timestamped.txt"
    save_transcript(result, output_file=txt_output, json_file=json_output, timestamped_file=timestamped_output)
    
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"{'='*80}")
    print(f"Audio file: {audio_file}")
    print(f"Transcript: {txt_output}")
    print(f"Timestamped transcript: {timestamped_output}")
    print(f"Detailed output: {json_output}")

if __name__ == "__main__":
    main()
