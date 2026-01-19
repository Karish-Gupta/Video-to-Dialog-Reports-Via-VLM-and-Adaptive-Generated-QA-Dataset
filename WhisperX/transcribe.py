import whisperx
import os
import json
import torch
import glob
import argparse
from typing import  Optional

def transcribe_audio_with_diarization(audio_file, model_size="base", device="cuda", compute_type="float16",
                                      hf_token=None):
    """Transcribe audio using WhisperX with speaker diarization"""

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        compute_type = "int8"

    print(f"\n{'=' * 80}")
    print(f"Step 1: Loading WhisperX model ({model_size}) on {device}")
    print(f"{'=' * 80}")

    # Load WhisperX model
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    # Load audio
    print(f"\nLoading audio file: {audio_file}")
    audio = whisperx.load_audio(audio_file)

    # Transcribe with WhisperX
    print(f"\n{'=' * 80}")
    print(f"Step 2: Transcribing audio")
    print(f"{'=' * 80}")
    result = model.transcribe(audio, batch_size=16)

    # Align whisper output
    print(f"\n{'=' * 80}")
    print(f"Step 3: Aligning transcription")
    print(f"{'=' * 80}")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # Perform speaker diarization with pyannote-audio (requires HF token)
    if hf_token:
        print(f"\n{'=' * 80}")
        print(f"Step 4: Performing speaker diarization with pyannote-audio")
        print(f"{'=' * 80}")
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
        print(f"\n{'=' * 80}")
        print("WARNING: Skipping speaker diarization (no HuggingFace token provided)")
        print(f"{'=' * 80}")

    return result


def save_transcript(result, output_file="transcript.txt", json_file="transcript.json",
                    timestamped_file="transcript_timestamped.txt"):
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
        f.write("=" * 80 + "\n")
        f.write("TRANSCRIPT WITH TIMESTAMPS\n")
        f.write("=" * 80 + "\n\n")

        if "segments" in result:
            for segment in result["segments"]:
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                speaker = segment.get("speaker", "Unknown")
                text = segment.get("text", "").strip()

                # Format timestamps as HH:MM:SS.mmm
                start_time = f"{int(start // 3600):02d}:{int((start % 3600) // 60):02d}:{start % 60:06.3f}"
                end_time = f"{int(end // 3600):02d}:{int((end % 3600) // 60):02d}:{end % 60:06.3f}"

                f.write(f"[{start_time} --> {end_time}] [{speaker}]\n")
                f.write(f"{text}\n\n")

    print(f"Human-readable timestamped transcript saved to: {timestamped_file}")

    # Print the transcript
    print("\n" + "=" * 80)
    print("TRANSCRIPT:")
    print("=" * 80)
    print(full_text)
    print("=" * 80)


def process_video_file(video_path: str, video_name: str, output_base_dir: str,
                       model_size: str = "base", device: str = "cuda",
                       compute_type: str = "float16", hf_token: Optional[str] = None):
    """
    Process a single video file: transcribe with diarization and save results.

    Args:
        video_path: Path to the video file
        video_name: Name of the video (without extension)
        output_base_dir: Base output directory
        model_size: WhisperX model size
        device: Device to use
        compute_type: Compute type for model
        hf_token: HuggingFace token for diarization
    """
    print(f"\n{'=' * 80}")
    print(f"PROCESSING: {video_name}")
    print(f"{'=' * 80}")

    # Create output directory for this video
    video_output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # Transcribe using WhisperX w/ diarization
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

    print(f"\n{'=' * 80}")
    print(f"COMPLETED: {video_name}")
    print(f"{'=' * 80}")
    print(f"Transcript: {txt_output}")
    print(f"JSON: {json_output}")
    print(f"Timestamped: {timestamped_output}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Transcribe videos with diarization and timestamps')
    parser.add_argument('--input_dir', type=str, default='data/videos',
                        help='Input directory containing video files (default: data/videos)')
    
    parser.add_argument('--output_dir', type=str, default='output_transcripts',
                        help='Output directory for transcripts (default: output)')
    
    parser.add_argument('--model_size', type=str, default='large-v3',
                        choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
                        help='WhisperX model size (default: large-v3)')
    
    parser.add_argument('--pattern', type=str, default='video*.mp4',
                        help='File pattern to match (default: video*.mp4)')
    
    parser.add_argument('--start_index', type=int, default=None,
                        help='Start processing from this video index (optional)')
    
    parser.add_argument('--end_index', type=int, default=None,
                        help='End processing at this video index (optional)')

    args = parser.parse_args()

    # Set device and compute type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Get HuggingFace token from env
    hf_token = os.environ.get("HF_TOKEN", None)

    print(f"\n{'=' * 80}")
    print(f"BATCH VIDEO TRANSCRIPTION")
    print(f"{'=' * 80}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model size: {args.model_size}")
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")
    print(f"File pattern: {args.pattern}")
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
        print(f"Processing videos {start} to {end - 1} ({len(video_files)} videos)")

    # Process each video
    successful = 0
    failed = 0

    for i, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        print(f"\n{'=' * 80}")
        print(f"Processing {i + 1}/{len(video_files)}: {video_name}")
        print(f"{'=' * 80}")

        try:
            process_video_file(
                video_path=video_path,
                video_name=video_name,
                output_base_dir=args.output_dir,
                model_size=args.model_size,
                device=device,
                compute_type=compute_type,
                hf_token=hf_token
            )
            successful += 1
        except Exception as e:
            print(f"\n{'=' * 80}")
            print(f"ERROR processing {video_name}: {str(e)}")
            print(f"{'=' * 80}")
            failed += 1
            continue

    # Final summary
    print(f"\n{'=' * 80}")
    print(f"BATCH PROCESSING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"Total videos: {len(video_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nAll transcripts saved with diarization and timestamps!")


if __name__ == "__main__":
    main()