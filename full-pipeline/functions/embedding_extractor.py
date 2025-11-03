import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2
from typing import List, Dict
import json
import pickle
import os
from datetime import datetime

from video_processor import create_video_chunks, extract_frames_from_chunk

def extract_clip_embeddings(frames: List[np.ndarray], model, processor, device: str) -> np.ndarray:
    """
    Extract CLIP image embeddings for a list of frames.
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

def process_video_with_embeddings(video_path: str, transcript: Dict, output_dir: str = "outputs2",
                                  chunk_duration: float = 5.0, frames_per_chunk: int = 5,
                                  model_name: str = "openai/clip-vit-base-patch32",
                                  device: str = "cuda") -> Dict:
    """
    Process video: chunk, sample frames, extract CLIP embeddings.
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
    
    # Create chunks (transcript may be empty; create_video_chunks handles missing segments)
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

def save_video_embeddings(result: Dict, output_dir: str = "outputs2", prefix: str = "video_embeddings"):
    """
    Save video processing results to JSON, numpy, and pickle files.
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