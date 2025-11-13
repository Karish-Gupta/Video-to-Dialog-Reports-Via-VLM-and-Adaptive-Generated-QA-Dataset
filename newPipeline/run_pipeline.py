import os
import torch
from datetime import datetime


from data_extraction.downloader import download_video, download_audio
from data_extraction.embedding_extractor import process_video_with_embeddings, save_video_embeddings
from data_extraction.utils import get_device, print_section, extract_transcript_chunks
from data_extraction.audio_transcriber import transcribe_audio_with_diarization, save_transcript
from model_inference.llm import *
from model_inference.vlm import *



def main():
    # YouTube video URL (HARDCODED AT THE MOMENT)
    youtube_url = "https://www.youtube.com/watch?v=83jt-xOJok4"

    # Output directory (changed to outputs2)
    output_dir = "outputs2"
    os.makedirs(output_dir, exist_ok=True)

    # Download full video
    video_file, video_info = download_video(youtube_url, output_path=output_dir)
    audio_file = download_audio(youtube_url, output_path=output_dir)
    print(f"\nVideo title: {video_info.get('title')}")
    print(f"Duration: {video_info.get('duration')} seconds")

    # Device selection
    device = 'cpu'  # Force CPU usage
    print(f"Using device: {'device'}")

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
        audio_file[0], 
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
    

    vlm_descriptions = []
    chunks = video_result['chunks']
    # Initialize LLM
    llm_model = "meta-llama/Llama-3.3-70B-Instruct"
    llm_ = llm(llm_model)

    vlm_model_name = "llava-hf/LLaVA-NeXT-Video-34B-hf"
    vlm_ = vlm(vlm_model_name)

    for i, chunk in enumerate(chunks):
        chunk_id = chunk['chunk_id']
        start_time = chunk['start_time']
        end_time = chunk['end_time']
        
        print(f"\n[{i+1}/{len(chunks)}] Processing chunk {chunk_id}: {start_time:.1f}s - {end_time:.1f}s")
        video_1_path = video_file


        # VLM summary
        vlm_conversation = vlm_.build_conversation()
        vlm_summary = vlm_.invoke(video_1_path, vlm_conversation)
        print(f"VLM Summary:\n{vlm_summary}")

        # Step 1 prompt
        chunk_transcript = extract_transcript_chunks(
            transcript_json_path=json_output,
            start_time=start_time,
            end_time=end_time,
            chunk_duration=chunk_duration)
        step_1_prompt = llm_.step_1_chat_template(chunk_transcript, vlm_summary)
        print(f"Step 1 Prompt:\n {step_1_prompt}")

        structured_output = llm_.invoke(step_1_prompt)
        print(f"Generated Structured Elements:\n {structured_output}")


        # Step 2 prompt
        step_2_prompt = llm_.step_2_chat_template(structured_output)
        print(f"Step 2 Prompt:\n {step_2_prompt}")

        generated_qs = llm_.invoke(step_2_prompt)
        print(f"Generated Questions:\n {generated_qs}")

        # Pass generated questions to VLM for answer generation
        qa_conversation = vlm_.build_qa_conversation(generated_qs)
        print (f"QA Prompt:\n {qa_conversation}")

        vlm_answers = vlm_.invoke(video_1_path, qa_conversation)
        print(f"VLM Generated Answers:\n {vlm_answers}") 


main()