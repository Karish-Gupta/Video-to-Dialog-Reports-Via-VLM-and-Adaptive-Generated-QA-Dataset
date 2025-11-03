import torch
import json
from typing import List, Dict, Tuple, Optional

def get_device():
    """Get available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")

def extract_transcript_chunks(
    transcript_json_path: str,
    start_time: float,
    end_time: Optional[float] = None,
    chunk_duration: float = 30.0,
    context_buffer: float = 5.0
) -> Dict:
    """
Example Usage: Extract 30-second chunk starting at 100 seconds
        chunk = extract_transcript_chunks("transcript.json", 100.0)
        
        # Extract specific time range
        chunk = extract_transcript_chunks("transcript.json", 100.0, 130.0)
        
        # Extract w/ context buffer (adds extra time before/after)
        chunk = extract_transcript_chunks("transcript.json", 100.0, 130.0, context_buffer=5.0)
    """
    # Load transcript JSON
    with open(transcript_json_path, 'r') as f:
        transcript_data = json.load(f)
    
    segments = transcript_data.get('segments', [])
    
    # Calculate end time if not provided
    if end_time is None:
        end_time = start_time + chunk_duration
    
    # Apply context buffer
    search_start = start_time - context_buffer
    search_end = end_time + context_buffer
    
    # Find all segments that start within the time range
    matching_segments = []
    for segment in segments:
        segment_start = segment.get('start', 0)
        
        # Check if segment starts within our time range
        if search_start <= segment_start <= search_end:
            matching_segments.append(segment)
    
    # Combine text from all matching segments
    combined_text = " ".join([seg.get('text', '').strip() for seg in matching_segments])
    
    # Get actual time range covered
    actual_start = matching_segments[0]['start'] if matching_segments else start_time
    actual_end = matching_segments[-1]['end'] if matching_segments else end_time
    
    result = {
        'segments': matching_segments,
        'start_time': actual_start,
        'end_time': actual_end,
        'requested_start': start_time,
        'requested_end': end_time,
        'text': combined_text,
        'num_segments': len(matching_segments),
        'chunk_duration': actual_end - actual_start if matching_segments else 0
    }
    
    return result
