import os
import json
from datetime import datetime
import glob
import argparse
import gc
from typing import Optional

# Patch torch.load to force weights_only=False 
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import whisperx


def load_models(model_size="base", device="cuda", compute_type="float16", hf_token=None):
    """Load all models once for batch processing"""
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        compute_type = "int8"
    
    print(f"\n{'='*80}")
    print(f"LOADING MODELS FOR BATCH (one-time setup)")
    print(f"{'='*80}")
    
    # Load WhisperX model
    print(f"Loading WhisperX model ({model_size}) on {device}...")
    whisper_model = whisperx.load_model(model_size, device, compute_type=compute_type)
    print("✓ WhisperX model loaded")
    
    # Load diarization model if token provided
    diarize_model = None
    if hf_token:
        print("Loading pyannote diarization pipeline...")
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token, 
            device=device
        )
        print("✓ Diarization model loaded")
    else:
        print("⚠ Skipping diarization model (no HF token)")
    
    print(f"{'='*80}")
    print("All models loaded and ready!")
    print(f"{'='*80}\n")
    
    return whisper_model, diarize_model, device


def transcribe_audio_with_models(audio_file, whisper_model, diarize_model, device="cuda"):
    """Transcribe audio using pre-loaded models (optimized for batch processing)"""
    
    # Load audio
    print(f"\nLoading audio file: {audio_file}")
    audio = whisperx.load_audio(audio_file)
    
    # Transcribe with WhisperX
    print(f"\n{'='*80}")
    print(f"Step 1: Transcribing audio")
    print(f"{'='*80}")
    result = whisper_model.transcribe(audio, batch_size=16)
    
    # Align whisper output
    print(f"\n{'='*80}")
    print(f"Step 2: Aligning transcription")
    print(f"{'='*80}")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Perform speaker diarization with pyannote-audio (if model provided)
    if diarize_model:
        print(f"\n{'='*80}")
        print(f"Step 3: Performing speaker diarization")
        print(f"{'='*80}")
        print("Running Voice Activity Detection (VAD) and speaker diarization...")
        diarize_segments = diarize_model(audio)
        
        # Assign speakers to words
        print("Assigning speakers to transcribed segments...")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        print("Speaker diarization completed successfully!")

        # Print speaker statistics
        speakers = set()
        for segment in result.get("segments", []):
            if "speaker" in segment:
                speakers.add(segment["speaker"])
        print(f"Detected {len(speakers)} unique speaker(s): {', '.join(sorted(speakers))}")
        
    else:
        print(f"\n{'='*80}")
        print("Skipping speaker diarization (no model loaded)")
        print(f"{'='*80}")
    
    return result


def transcribe_audio_with_diarization(audio_file, model_size="base", device="cuda", compute_type="float16", hf_token=None):
    """Transcribe audio using WhisperX with speaker diarization (loads models each time - use for single videos)"""
    
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
    
    # Perform speaker diarization with pyannote-audio (requires HF token)
    if hf_token:
        print(f"\n{'='*80}")
        print(f"Step 4: Performing speaker diarization with pyannote-audio")
        print(f"{'='*80}")
        print("Loading pyannote diarization pipeline with VAD...")
        
        # Load diarization pipeline with VAD
        # This uses pyannote-audio which includes Voice Activity Detection
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token, 
            device=device
        )
        
        # Perform diarization - pyannote automatically performs VAD first
        print("Running Voice Activity Detection (VAD) and speaker diarization...")
        diarize_segments = diarize_model(audio)
        
        # Assign speakers to words
        print("Assigning speakers to transcribed segments...")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        print("Speaker diarization completed successfully!")

        # Print speaker statistics
        speakers = set()
        for segment in result.get("segments", []):
            if "speaker" in segment:
                speakers.add(segment["speaker"])
        print(f"Detected {len(speakers)} unique speaker(s): {', '.join(sorted(speakers))}")
        
    else:
        print(f"\n{'='*80}")
        print("WARNING: Skipping speaker diarization (no HuggingFace token provided)")
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


def process_video_file(video_path: str, video_name: str, output_base_dir: str, 
                       whisper_model=None, diarize_model=None, device: str = "cuda",
                       model_size: str = "base", compute_type: str = "float16", hf_token: Optional[str] = None):
    """
    Process a single video file: transcribe with diarization and save results.
    
    Args:
        video_path: Path to the video file
        video_name: Name of the video (without extension)
        output_base_dir: Base output directory
        whisper_model: Pre-loaded WhisperX model (if None, will load new one)
        diarize_model: Pre-loaded diarization model (if None, will load new one if hf_token provided)
        device: Device to use
        model_size: WhisperX model size (only used if whisper_model is None)
        compute_type: Compute type for model (only used if whisper_model is None)
        hf_token: HuggingFace token for diarization (only used if diarize_model is None)
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {video_name}")
    print(f"{'='*80}")
    
    # Create output directory for this video
    video_output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Transcribe using pre-loaded models or load new ones
    if whisper_model is not None:
        # Use pre-loaded models (batch mode - optimized)
        result = transcribe_audio_with_models(
            video_path, 
            whisper_model=whisper_model,
            diarize_model=diarize_model,
            device=device
        )
    else:
        # Load models fresh (single video mode)
        result = transcribe_audio_with_diarization(
            video_path, 
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            hf_token=hf_token
        )
    
    # Format output filename: video1 -> Video1Transcript
    formatted_name = video_name.replace('video', 'Video') + 'Transcript'
    
    txt_output = os.path.join(video_output_dir, f"{formatted_name}.txt")
    json_output = os.path.join(video_output_dir, f"{formatted_name}.json")
    timestamped_output = os.path.join(video_output_dir, f"{formatted_name}_timestamped.txt")
    
    save_transcript(result, output_file=txt_output, json_file=json_output, timestamped_file=timestamped_output)
    
    print(f"\n{'='*80}")
    print(f"COMPLETED: {video_name}")
    print(f"{'='*80}")
    print(f"Transcript: {txt_output}")
    print(f"JSON: {json_output}")
    print(f"Timestamped: {timestamped_output}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Transcribe videos with diarization and timestamps')
    parser.add_argument('--input_dir', type=str, default='output_segments',
                       help='Input directory containing video files (default: output_segments)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for transcripts (default: output)')
    parser.add_argument('--model_size', type=str, default='large-v2',
                       choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
                       help='WhisperX model size (default: large-v2)')
    parser.add_argument('--pattern', type=str, default='video*.mp4',
                       help='File pattern to match (default: video*.mp4)')
    parser.add_argument('--start_index', type=int, default=None,
                       help='Start processing from this video index (optional)')
    parser.add_argument('--end_index', type=int, default=None,
                       help='End processing at this video index (optional)')
    parser.add_argument('--batch_size', type=int, default=15,
                       help='Number of videos to process in each batch (default: 15)')
    parser.add_argument('--no_auto_continue', action='store_true',
                       help='Prompt before each batch (default: auto-continue enabled)')
    
    args = parser.parse_args()
    
    # Set auto_continue based on flag (default is True, set to False if flag is present)
    args.auto_continue = not args.no_auto_continue
    
    # Set device and compute type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    # Get HuggingFace token from env
    hf_token = None  # Disabled diarization
    # hf_token = os.environ.get("HF_TOKEN")
    
    print(f"\n{'='*80}")
    print(f"BATCH VIDEO TRANSCRIPTION")
    print(f"{'='*80}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model size: {args.model_size}")
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")
    print(f"File pattern: {args.pattern}")
    print(f"Batch size: {args.batch_size}")
    if hf_token:
        print("HuggingFace token found - speaker diarization will be enabled")
    else:
        print("WARNING: No HuggingFace token - speaker diarization will be skipped")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all video files
    video_pattern = os.path.join(args.input_dir, args.pattern)
    video_files = sorted(glob.glob(video_pattern))
    
    if not video_files:
        print(f"\nERROR: No video files found matching pattern: {video_pattern}")
        return
    
    print(f"\nFound {len(video_files)} video files")
    
    # Filter by index range if specified
    if args.start_index is not None or args.end_index is not None:
        start = args.start_index if args.start_index is not None else 0
        end = args.end_index if args.end_index is not None else len(video_files)
        video_files = video_files[start:end]
        print(f"Processing videos {start} to {end-1} ({len(video_files)} videos)")
    
    # Calculate number of batches
    total_videos = len(video_files)
    num_batches = (total_videos + args.batch_size - 1) // args.batch_size  # Ceiling division
    
    print(f"\nTotal videos to process: {total_videos}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {num_batches}")
    
    # Process videos in batches
    total_successful = 0
    total_failed = 0
    
    for batch_num in range(num_batches):
        batch_start = batch_num * args.batch_size
        batch_end = min(batch_start + args.batch_size, total_videos)
        batch_videos = video_files[batch_start:batch_end]
        
        print(f"\n{'#'*80}")
        print(f"# BATCH {batch_num + 1}/{num_batches}")
        print(f"# Processing videos {batch_start} to {batch_end - 1}")
        print(f"# ({len(batch_videos)} videos in this batch)")
        print(f"{'#'*80}")
        
        # Load models once for this batch
        whisper_model, diarize_model, device = load_models(
            model_size=args.model_size,
            device=device,
            compute_type=compute_type,
            hf_token=hf_token
        )
        
        # Process each video in this batch
        batch_successful = 0
        batch_failed = 0
        
        for i, video_path in enumerate(batch_videos):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            global_index = batch_start + i
            
            print(f"\n{'='*80}")
            print(f"Processing {global_index + 1}/{total_videos} (Batch {batch_num + 1}, Video {i + 1}/{len(batch_videos)}): {video_name}")
            print(f"{'='*80}")
            
            try:
                process_video_file(
                    video_path=video_path,
                    video_name=video_name,
                    output_base_dir=args.output_dir,
                    whisper_model=whisper_model,
                    diarize_model=diarize_model,
                    device=device
                )
                batch_successful += 1
                total_successful += 1
            except Exception as e:
                print(f"\n{'='*80}")
                print(f"ERROR processing {video_name}: {str(e)}")
                print(f"{'='*80}")
                batch_failed += 1
                total_failed += 1
                continue
        
        # Batch summary
        print(f"\n{'#'*80}")
        print(f"# BATCH {batch_num + 1}/{num_batches} COMPLETE!")
        print(f"{'#'*80}")
        print(f"Batch {batch_num + 1} - Successful: {batch_successful}, Failed: {batch_failed}")
        print(f"Overall Progress - Successful: {total_successful}, Failed: {total_failed}, Remaining: {total_videos - batch_end}")
        print(f"{'#'*80}")
        
        # Clean up memory after batch
        print("Cleaning up memory...")
        del whisper_model
        del diarize_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Memory cleanup complete.")
        
        # Ask to continue to next batch (unless it's the last batch or auto_continue is set)
        if batch_end < total_videos and not args.auto_continue:
            print(f"\n{'='*80}")
            print(f"PAUSING AFTER BATCH {batch_num + 1}")
            print(f"{'='*80}")
            print(f"Processed {batch_end}/{total_videos} videos so far")
            print(f"Next batch will process videos {batch_end} to {min(batch_end + args.batch_size, total_videos) - 1}")
            
            response = input("\nContinue to next batch? (y/n): ").strip().lower()
            if response != 'y':
                print("\nStopping batch processing.")
                print(f"To resume later, run with: --start_index {batch_end}")
                break
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total videos processed: {batch_end if batch_end < total_videos and not args.auto_continue else total_videos}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nAll transcripts saved with diarization and timestamps!")

if __name__ == "__main__":
    main()
