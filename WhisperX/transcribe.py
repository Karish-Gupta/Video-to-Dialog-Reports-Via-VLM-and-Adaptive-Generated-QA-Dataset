import whisperx
import yt_dlp
import os
import json
from datetime import datetime
import torch
import cv2
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

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
    
    # Perform speaker diarization with pyannote-audio (requires HF token)
    if hf_token:
        print(f"\n{'='*80}")
        print(f"Step 4: Performing speaker diarization with pyannote-audio")
        print(f"{'='*80}")
        print("Loading pyannote diarization pipeline with VAD...")
        
        # Load diarization pipeline with VAD
        # This uses pyannote-audio which includes Voice Activity Detection
        diarize_model = whisperx.DiarizationPipeline(
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


def download_video(youtube_url, output_path="output"):
    """Download video from YouTube"""
    print(f"Downloading video from: {youtube_url}")
    
    os.makedirs(output_path, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'quiet': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_file = ydl.prepare_filename(info)
    
    print(f"Video downloaded: {video_file}")
    return video_file, info


def create_video_chunks(video_duration: float, transcript: Dict, chunk_duration: float = 5.0) -> List[Dict]:
    """
    Create naive temporal chunks of fixed duration aligned with transcript.
    
    Args:
        video_duration: Total duration of video in seconds
        transcript: Transcript dictionary with segments
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []
    current_time = 0.0
    chunk_id = 0
    
    print(f"\n{'='*80}")
    print(f"Creating video chunks with duration: {chunk_duration}s")
    print(f"Video duration: {video_duration:.2f}s")
    print(f"{'='*80}")
    
    while current_time < video_duration:
        end_time = min(current_time + chunk_duration, video_duration)
        
        chunk = {
            'chunk_id': chunk_id,
            'start_time': current_time,
            'end_time': end_time,
            'duration': end_time - current_time,
            'transcript_segments': []
        }
        
        # Find transcript segments that overlap with this chunk
        if 'segments' in transcript:
            for segment in transcript['segments']:
                seg_start = segment.get('start', 0)
                seg_end = segment.get('end', 0)
                
                # Check if segment overlaps with chunk
                if seg_start < end_time and seg_end > current_time:
                    chunk['transcript_segments'].append({
                        'text': segment.get('text', '').strip(),
                        'start': seg_start,
                        'end': seg_end,
                        'speaker': segment.get('speaker', 'Unknown'),
                        'words': segment.get('words', [])
                    })
        
        chunks.append(chunk)
        current_time = end_time
        chunk_id += 1
    
    print(f"Created {len(chunks)} chunks")
    return chunks


def extract_frames_from_chunk(video_path: str, start_time: float, end_time: float, 
                               frames_per_chunk: int = 5) -> List[np.ndarray]:
    """
    Extract uniformly sampled frames from a video chunk.
    
    Args:
        video_path: Path to video file
        start_time: Start time of chunk in seconds
        end_time: End time of chunk in seconds
        frames_per_chunk: Number of frames to sample
        
    Returns:
        List of frames as numpy arrays (RGB format)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame indices to sample
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame
    
    if total_frames <= 0:
        cap.release()
        return []
    
    # Sample frames uniformly
    frame_indices = np.linspace(start_frame, end_frame - 1, 
                               min(frames_per_chunk, total_frames), 
                               dtype=int)
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames


def extract_clip_embeddings(frames: List[np.ndarray], model, processor, device: str) -> np.ndarray:
    """
    Extract CLIP image embeddings for a list of frames.
    
    Args:
        frames: List of frames as numpy arrays (RGB format)
        model: CLIP model
        processor: CLIP processor
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        Numpy array of embeddings, shape (num_frames, embedding_dim)
    """
    if not frames:
        return np.array([])
    
    # Convert numpy arrays to PIL Images
    pil_images = [Image.fromarray(frame) for frame in frames]
    
    # Process images
    inputs = processor(images=pil_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract embeddings
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # Normalize embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy
    embeddings = image_features.cpu().numpy()
    return embeddings


def process_video_with_embeddings(video_path: str, transcript: Dict, output_dir: str = "output",
                                  chunk_duration: float = 5.0, frames_per_chunk: int = 5,
                                  model_name: str = "openai/clip-vit-base-patch32",
                                  device: str = "cuda") -> Dict:
    """
    Process video: chunk, sample frames, extract CLIP embeddings.
    
    Args:
        video_path: Path to video file
        transcript: Transcript dictionary
        output_dir: Output directory
        chunk_duration: Duration of each chunk in seconds
        frames_per_chunk: Number of frames to sample per chunk
        model_name: CLIP model name
        device: Device to run on
        
    Returns:
        Dictionary containing all processed data
    """
    print(f"\n{'='*80}")
    print("PROCESSING VIDEO WITH CLIP EMBEDDINGS")
    print(f"{'='*80}\n")
    
    # Load CLIP model
    print(f"Loading CLIP model: {model_name} on {device}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    print("CLIP model loaded successfully!")
    
    # Get video metadata
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps
    cap.release()
    
    print(f"\nVideo: {video_path}")
    print(f"Duration: {video_duration:.2f}s, FPS: {fps:.2f}, Frames: {frame_count}")
    
    # Create chunks
    chunks = create_video_chunks(video_duration, transcript, chunk_duration)
    
    # Process each chunk
    all_embeddings = []
    processed_chunks = []
    
    print(f"\n{'='*80}")
    print("EXTRACTING FRAMES AND EMBEDDINGS")
    print(f"{'='*80}\n")
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}: "
              f"{chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s")
        
        # Extract frames
        frames = extract_frames_from_chunk(
            video_path, 
            chunk['start_time'], 
            chunk['end_time'],
            frames_per_chunk
        )
        
        print(f"  Extracted {len(frames)} frames")
        
        # Extract embeddings
        if frames:
            embeddings = extract_clip_embeddings(frames, model, processor, device)
            print(f"  Extracted embeddings: shape {embeddings.shape}")
            
            # Store embeddings
            all_embeddings.append(embeddings)
            
            # Add embedding info to chunk
            chunk['num_frames'] = len(frames)
            chunk['embedding_shape'] = embeddings.shape
            chunk['frame_timestamps'] = [
                chunk['start_time'] + (chunk['duration'] / len(frames)) * j 
                for j in range(len(frames))
            ]
        else:
            chunk['num_frames'] = 0
            chunk['embedding_shape'] = (0, 0)
            chunk['frame_timestamps'] = []
        
        processed_chunks.append(chunk)
    
    # Combine all embeddings
    if all_embeddings:
        combined_embeddings = np.vstack(all_embeddings)
    else:
        combined_embeddings = np.array([])
    
    print(f"\n{'='*80}")
    print("VIDEO PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total chunks: {len(processed_chunks)}")
    print(f"Total frames: {sum(c['num_frames'] for c in processed_chunks)}")
    print(f"Total embeddings shape: {combined_embeddings.shape}")
    
    # Create result structure
    result = {
        'video_info': {
            'path': video_path,
            'duration': video_duration,
            'fps': fps,
            'frame_count': frame_count
        },
        'processing_params': {
            'chunk_duration': chunk_duration,
            'frames_per_chunk': frames_per_chunk,
            'model_name': model_name,
            'embedding_dim': combined_embeddings.shape[-1] if combined_embeddings.size > 0 else 0
        },
        'chunks': processed_chunks,
        'embeddings': combined_embeddings,
        'timestamp': datetime.now().isoformat()
    }
    
    return result


def save_video_embeddings(result: Dict, output_dir: str = "output", prefix: str = "video_embeddings"):
    """
    Save video processing results to JSON, numpy, and pickle files.
    
    Args:
        result: Result dictionary from process_video_with_embeddings
        output_dir: Directory to save outputs
        prefix: Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare JSON-serializable version (without embeddings)
    json_result = {
        'video_info': result['video_info'],
        'processing_params': result['processing_params'],
        'timestamp': result['timestamp'],
        'chunks': []
    }
    
    # Clean chunks for JSON
    for chunk in result['chunks']:
        chunk_copy = chunk.copy()
        if 'embedding_shape' in chunk_copy:
            chunk_copy['embedding_shape'] = list(chunk_copy['embedding_shape'])
        json_result['chunks'].append(chunk_copy)
    
    # Save human-readable JSON
    json_path = os.path.join(output_dir, f"{prefix}_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*80}")
    print("SAVED OUTPUT FILES")
    print(f"{'='*80}")
    print(f"Metadata (JSON): {json_path}")
    
    # Save embeddings as numpy array
    embeddings = result['embeddings']
    if embeddings.size > 0:
        npy_path = os.path.join(output_dir, f"{prefix}_{timestamp}_embeddings.npy")
        np.save(npy_path, embeddings)
        print(f"Embeddings (numpy): {npy_path}")
    
    # Save complete result as pkl
    pickle_path = os.path.join(output_dir, f"{prefix}_{timestamp}_complete.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Complete data (pickle): {pickle_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total chunks: {len(result['chunks'])}")
    print(f"Total frames: {sum(c['num_frames'] for c in result['chunks'])}")
    print(f"Embedding shape: {embeddings.shape if embeddings.size > 0 else 'N/A'}")
    print(f"Embedding dimension: {result['processing_params']['embedding_dim']}")
    
    # Print example chunks with transcript
    print(f"\n{'='*80}")
    print("EXAMPLE CHUNKS (first 3)")
    print(f"{'='*80}")
    for i, chunk in enumerate(result['chunks'][:3]):
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"  Time: {chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s")
        print(f"  Frames: {chunk['num_frames']}")
        print(f"  Transcript segments: {len(chunk['transcript_segments'])}")
        if chunk['transcript_segments']:
            for seg in chunk['transcript_segments'][:2]:  # Show first 2 segments
                speaker = seg.get('speaker', 'Unknown')
                text = seg['text'][:100] + ('...' if len(seg['text']) > 100 else '')
                print(f"    [{speaker}]: {text}")


def main():
    # YouTube video URL (HARDCODED AT THE MOMENT)
    youtube_url = "https://www.youtube.com/watch?v=aNXf82gWb5A"
    
    # Create output dir
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download audio for transcription
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
    print(f"TRANSCRIPTION COMPLETE!")
    print(f"{'='*80}")
    print(f"Audio file: {audio_file}")
    print(f"Transcript: {txt_output}")
    print(f"Timestamped transcript: {timestamped_output}")
    print(f"Detailed output: {json_output}")
    
    # =========================================================================
    # VIDEO PROCESSING WITH CLIP EMBEDDINGS
    # =========================================================================
    
    # Download video (full video, not just audio)
    print(f"\n{'='*80}")
    print("STARTING VIDEO PROCESSING")
    print(f"{'='*80}")
    
    video_file, _ = download_video(youtube_url, output_path=output_dir)
    
    # Process video with embeddings
    # Parameters for naive chunking
    chunk_duration = 5.0  # seconds
    frames_per_chunk = 5  # frames to sample per chunk
    clip_model = "openai/clip-vit-base-patch32"
    
    print(f"\nVideo processing parameters:")
    print(f"  Chunk duration: {chunk_duration}s")
    print(f"  Frames per chunk: {frames_per_chunk}")
    print(f"  CLIP model: {clip_model}")
    
    video_result = process_video_with_embeddings(
        video_path=video_file,
        transcript=result,  # Use the transcript we just created
        output_dir=output_dir,
        chunk_duration=chunk_duration,
        frames_per_chunk=frames_per_chunk,
        model_name=clip_model,
        device=device
    )
    
    # Save video embeddings and metadata
    save_video_embeddings(video_result, output_dir=output_dir, prefix="video_embeddings")
    
    print(f"\n{'='*80}")
    print(f"ALL PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  - Transcript (text): {txt_output}")
    print(f"  - Transcript (timestamped): {timestamped_output}")
    print(f"  - Transcript (JSON): {json_output}")
    print(f"  - Video embeddings (JSON, .npy, .pkl) in {output_dir}/")
    print(f"\nThe dataset is ready for semantic search and QA tasks!")

if __name__ == "__main__":
    main()
