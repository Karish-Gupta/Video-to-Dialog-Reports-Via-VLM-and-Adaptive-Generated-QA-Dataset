import os
import torch
from functions.downloader import download_video
from functions.embedding_extractor import process_video_with_embeddings, save_video_embeddings
from functions.vlm_processor import load_llava_model, process_chunk_with_vlm, save_vlm_descriptions
from functions.video_processor import extract_video_chunk_with_decord
from functions.utils import get_device, print_section

def main():
    # YouTube video URL (HARDCODED AT THE MOMENT)
    youtube_url = "https://www.youtube.com/watch?v=83jt-xOJok4"

    # Output directory (changed to outputs2)
    output_dir = "outputs2"
    os.makedirs(output_dir, exist_ok=True)

    # Download full video
    video_file, video_info = download_video(youtube_url, output_path=output_dir)
    print(f"\nVideo title: {video_info.get('title')}")
    print(f"Duration: {video_info.get('duration')} seconds")

    # Device selection
    device = get_device()
    print(f"Using device: {device}")

    # =========================================================================
    # STEP 1: Process video with CLIP embeddings (30-second chunks)
    # =========================================================================
    chunk_duration = 30.0  
    frames_per_chunk = 64
    clip_model = "openai/clip-vit-base-patch32"

    print_section("STEP 1: EXTRACTING CLIP EMBEDDINGS")
    print(f"Video processing parameters:")
    print(f"  Chunk duration: {chunk_duration}s")
    print(f"  Frames per chunk (for CLIP): {frames_per_chunk}")
    print(f"  CLIP model: {clip_model}")

    # Pass empty transcript (no WhisperX transcription in this simplified flow)
    video_result = process_video_with_embeddings(
        video_path=video_file,
        transcript={},
        output_dir=output_dir,
        chunk_duration=chunk_duration,
        frames_per_chunk=frames_per_chunk,
        model_name=clip_model,
        device=device
    )

    # Save CLIP embeddings to outputs2
    save_video_embeddings(video_result, output_dir=output_dir, prefix="video_embeddings")

    # =========================================================================
    # STEP 2: Process each 30-second chunk with LLaVA-NeXT-Video
    # =========================================================================
    print_section("STEP 2: PROCESSING CHUNKS WITH LLaVA-NeXT-VIDEO")
    
    # Load LLaVA model
    vlm_model, vlm_processor = load_llava_model(device=device)
    
    # Process each chunk with VLM
    vlm_descriptions = []
    chunks = video_result['chunks']
    
    print(f"\nProcessing {len(chunks)} chunks with VLM...")
    
    for i, chunk in enumerate(chunks):
        chunk_id = chunk['chunk_id']
        start_time = chunk['start_time']
        end_time = chunk['end_time']
        
        print(f"\n[{i+1}/{len(chunks)}] Processing chunk {chunk_id}: {start_time:.1f}s - {end_time:.1f}s")
        
        # Extract frames for VLM (using decord, up to 32 frames)
        frames = extract_video_chunk_with_decord(
            video_file, 
            start_time, 
            end_time, 
            max_frames=32
        )
        
        if frames:
            print(f"  Extracted {len(frames)} frames for VLM processing")
            
            # Process with VLM
            description = process_chunk_with_vlm(
                frames, 
                vlm_model, 
                vlm_processor,
                prompt="Describe what happens in this video segment in detail."
            )
            
            print(f"  Description: {description[:100]}..." if len(description) > 100 else f"  Description: {description}")
        else:
            description = "[No frames extracted for this chunk]"
            print(f"  Warning: No frames extracted")
        
        vlm_descriptions.append({
            'chunk_id': chunk_id,
            'start_time': start_time,
            'end_time': end_time,
            'description': description
        })
    
    # Save VLM descriptions to timestamped text file
    vlm_output_path = save_vlm_descriptions(vlm_descriptions, output_dir=output_dir, prefix="vlm_descriptions")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("ALL PROCESSING COMPLETE!")
    print(f"Generated files in: {output_dir}/")
    print("- Video metadata and chunk info (JSON)")
    print("- CLIP embeddings (.npy) and complete pickle (.pkl)")
    print(f"- VLM descriptions: {vlm_output_path}")
    print(f"\nProcessed {len(chunks)} chunks of {chunk_duration}s each")

if __name__ == "__main__":
    main()