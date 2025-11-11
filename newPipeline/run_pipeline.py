import os
import sys
import argparse
import json
from typing import List, Dict

# Ensure repo root on path so package-style imports work when run from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from newPipeline.data_extraction.downloader import download_video, download_audio
from newPipeline.data_extraction.audio_transcriber import transcribe_audio_with_diarization, save_transcript
from newPipeline.data_extraction.video_processor import create_video_chunks, extract_video_chunk_with_decord
from newPipeline.model_inference.llm import llm
from newPipeline.model_inference.vlm import vlm
from decord import VideoReader, cpu


def get_video_duration(video_path: str) -> float:
    vr = VideoReader(video_path, ctx=cpu())
    fps = vr.get_avg_fps()
    duration = len(vr) / float(fps) if fps and fps > 0 else 0.0
    return duration


def concat_transcript_segments(segments: List[Dict]) -> str:
    texts = []
    for s in segments:
        t = s.get('text', '').strip()
        if t:
            texts.append(t)
    return '\n'.join(texts)


def run_pipeline(youtube_url: str,
                 out_dir: str = 'outputs',
                 chunk_duration: float = 5.0,
                 llm_model_name: str = 'meta-llama/Llama-3.3-70B-Instruct',
                 vlm_model_name: str = 'llava-hf/LLaVA-NeXT-Video-34B-hf',
                 max_frames_per_chunk: int = 32):

    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading video and audio from: {youtube_url}")
    video_file = download_video(youtube_url, output_path=out_dir)
    audio_file = download_audio(youtube_url, output_path=out_dir)

    print(f"Video saved to: {video_file}")
    print(f"Audio saved to: {audio_file}")

    # Transcribe (use HF token from env if present)
    hf_token = os.environ.get('HF_TOKEN')
    transcript_result = transcribe_audio_with_diarization(audio_file,
                                                         model_size='base',
                                                         device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', None) or os.getenv('CUDA') else 'cpu',
                                                         hf_token=hf_token)

    # Save transcripts to out_dir
    transcript_txt = os.path.join(out_dir, 'transcript.txt')
    transcript_json = os.path.join(out_dir, 'transcript.json')
    transcript_timestamp = os.path.join(out_dir, 'transcript_timestamped.txt')
    save_transcript(transcript_result, output_file=transcript_txt, json_file=transcript_json, timestamped_file=transcript_timestamp)

    # Video duration and chunking
    video_duration = get_video_duration(video_file)
    chunks = create_video_chunks(video_duration, transcript_result, chunk_duration=chunk_duration)

    # Initialize models once
    print("Initializing models (this may take a while)...")
    vlm_model = vlm(vlm_model_name)
    llm_model = llm(llm_model_name)

    results = []

    for chunk in chunks:
        cid = chunk['chunk_id']
        start = chunk['start_time']
        end = chunk['end_time']

        print(f"\n--- Processing chunk {cid}: {start:.2f}s -> {end:.2f}s ({chunk['duration']:.2f}s) ---")

        # Extract frames for this chunk
        frames = extract_video_chunk_with_decord(video_file, start, end, max_frames=max_frames_per_chunk)
        if not frames:
            print(f"No frames found for chunk {cid}, skipping.")
            continue

        # Build chunk-level transcript
        chunk_transcript = concat_transcript_segments(chunk.get('transcript_segments', []))

        # VLM summary for chunk (use processor + model directly with the chunk frames)
        conversation = vlm_model.build_conversation()
        processed_text = vlm_model.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = vlm_model.processor(
            text=[processed_text],
            videos=[frames],
            padding=True,
            return_tensors='pt'
        ).to(vlm_model.model.device)

        out = vlm_model.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
        vlm_summary = vlm_model.processor.batch_decode(out, skip_special_tokens=True)[0]
        print(f"VLM summary (chunk {cid}): {vlm_summary}")

        # LLM Step 1: structured extraction
        step1_prompt = llm_model.step_1_chat_template(chunk_transcript, vlm_summary)
        structured_output = llm_model.invoke(step1_prompt)
        print(f"Structured output (chunk {cid}): {structured_output}")

        # LLM Step 2: generate questions
        step2_prompt = llm_model.step_2_chat_template(structured_output)
        generated_questions = llm_model.invoke(step2_prompt)
        print(f"Generated questions (chunk {cid}):\n{generated_questions}")

        # VLM answer generation for those questions
        qa_conversation = vlm_model.build_qa_conversation(generated_questions)
        processed_qa_text = vlm_model.processor.apply_chat_template(qa_conversation, add_generation_prompt=True)
        qa_inputs = vlm_model.processor(
            text=[processed_qa_text],
            videos=[frames],
            padding=True,
            return_tensors='pt'
        ).to(vlm_model.model.device)

        qa_out = vlm_model.model.generate(
            **qa_inputs,
            max_new_tokens=512,
            do_sample=False
        )
        vlm_answers = vlm_model.processor.batch_decode(qa_out, skip_special_tokens=True)[0]
        print(f"VLM answers (chunk {cid}):\n{vlm_answers}")

        # Collect results
        item = {
            'chunk_id': cid,
            'start': start,
            'end': end,
            'vlm_summary': vlm_summary,
            'structured_output': structured_output,
            'generated_questions': generated_questions,
            'vlm_answers': vlm_answers
        }
        results.append(item)

    # Write results to file
    out_results_path = os.path.join(out_dir, 'chunk_inference_results.json')
    with open(out_results_path, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\nAll done. Results saved to: {out_results_path}")


def cli():
    p = argparse.ArgumentParser(description='Run end-to-end VLM+LLM pipeline on a YouTube URL and per-chunk inference')
    p.add_argument('youtube_url', type=str, help='YouTube video URL to process')
    p.add_argument('--out', type=str, default='outputs', help='Output folder')
    p.add_argument('--chunk-duration', type=float, default=60.0, help='Seconds per chunk')
    p.add_argument('--llm-model', type=str, default='meta-llama/Llama-3.3-70B-Instruct', help='LLM model name')
    p.add_argument('--vlm-model', type=str, default='llava-hf/LLaVA-NeXT-Video-34B-hf', help='VLM model name')
    args = p.parse_args()

    run_pipeline(args.youtube_url, out_dir=args.out, chunk_duration=args.chunk_duration,
                 llm_model_name=args.llm_model, vlm_model_name=args.vlm_model)


if __name__ == '__main__':
    cli()
