import cv2
import numpy as np
from typing import List, Dict
from decord import VideoReader, cpu

def create_video_chunks(video_duration: float, transcript: Dict, chunk_duration: float = 5.0) -> List[Dict]:
    """
    Create naive temporal chunks of fixed duration aligned with transcript.
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

def extract_video_chunk_with_decord(video_path: str, start_time: float, end_time: float, 
                                     max_frames: int = 32) -> List[np.ndarray]:
    """
    Extract frames from a video chunk using decord for VLM processing.
    """
    vr = VideoReader(video_path, ctx=cpu())
    fps = vr.get_avg_fps()
    
    # Calculate frame indices for the chunk
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Ensure we don't go beyond video bounds
    start_frame = max(0, start_frame)
    end_frame = min(len(vr), end_frame)
    
    if start_frame >= end_frame:
        return []
    
    # Sample frames uniformly from the chunk
    num_frames = min(max_frames, end_frame - start_frame)
    indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    
    frames = [vr[i].asnumpy() for i in indices]
    return frames