import whisperx
import glob
import os
import json
from datetime import datetime
import torch

def transcribe_audio_with_diarization(audio_file, model_size="base", device="cuda", compute_type="float16", hf_token=None):
    """Transcribe audio using WhisperX with speaker diarization"""
    
    # Check if GPU is available
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
    # Process local MP4/audio clips 
    clips_dir = os.path.join(os.path.dirname(__file__), "60SecClips")
    if not os.path.isdir(clips_dir):
        print(f"Clips directory not found: {clips_dir}")
        return

    exts = ("*.mp4")
    clips = []
    for e in exts:
        clips.extend(glob.glob(os.path.join(clips_dir, e)))
    clips = sorted(clips)
    # Remove any non-file entries (defensive)
    clips = [p for p in clips if os.path.isfile(p)]

    if not clips:
        print(f"No valid clip files found in {clips_dir} after filtering. Check file extensions.")
        return

    if not clips:
        print(f"No clips found in {clips_dir}. Place clip files and retry.")
        return

    # Print a concise header listing all clips to be processed
    print("\n" + "="*100)
    print(f"Found {len(clips)} clip(s) to process:")
    for i, c in enumerate(clips, 1):
        print(f"  {i}. {os.path.basename(c)}")
    print("="*100 + "\n")

    # Transcription settings
    model_size = "base"  # choose: tiny, base, small, medium, large-v2, large-v3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    hf_token = os.environ.get("HF_TOKEN", None)

    print(f"Using device: {device}")
    print(f"Compute type: {compute_type}")
    if hf_token:
        print("HuggingFace token found - speaker diarization will be enabled")
    else:
        print("No HuggingFace token - speaker diarization will be skipped")

    # Ensure outputs go under the WhisperX package folder
    top_output = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(top_output, exist_ok=True)

    for clip_idx, clip_path in enumerate(clips, 1):
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        print("\n" + "="*100)
        print(f"Processing clip {clip_idx}/{len(clips)}: {clip_path}")
        print("="*100 + "\n")

        output_dir = os.path.join(top_output, clip_name)
        os.makedirs(output_dir, exist_ok=True)

        result = transcribe_audio_with_diarization(
            clip_path,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            hf_token=hf_token
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_output = f"{output_dir}/transcript_{clip_name}_{timestamp}.txt"
        json_output = f"{output_dir}/transcript_{clip_name}_{timestamp}.json"
        timestamped_output = f"{output_dir}/transcript_{clip_name}_{timestamp}_timestamped.txt"
        save_transcript(result, output_file=txt_output, json_file=json_output, timestamped_file=timestamped_output)

        print("\n" + "="*100)
        print(f"Finished clip: {clip_name} ({clip_idx}/{len(clips)})")
        print(f"Outputs:\n  - {txt_output}\n  - {timestamped_output}\n  - {json_output}")
        print("="*100 + "\n")

if __name__ == "__main__":
    main()