import json
import argparse
import os
import re
from pathlib import Path
from models import gemini_model
from models.gemini_model import *
from models.sft_question_generation_model import QuestionGenerationModelSFT


BASE_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"  
ADAPTER_DIR = "./qwen3-30b-instruct-police-questions-lora-gemini-vlm"
VIDEO_DIR = "pipeline/eval_videos"  # Directory where videos are stored


vlm_model_name = "gemini-2.5-flash"
gemini = gemini_model(vlm_model_name)
# question generation model (Qwen-based)
qwen_model = QuestionGenerationModelSFT(BASE_MODEL, ADAPTER_DIR)



def process_json_file(input_file, output_file, num_examples=5):
    """Process a JSON file (array of examples) and generate outputs."""
    # qwen_model is initialized globally
    results = []
    count = 0
    captions_dir = "pipeline/baseline_captions"
    os.makedirs(captions_dir, exist_ok=True)
    
    # load entire JSON array
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            raw = json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file {input_file}: {e}")
            return

    try:
        examples = qwen_model.normalize_examples(raw, max_examples=num_examples)
    except Exception as e:
        print(f"Error normalizing input examples: {e}")
        return

    for example in examples:
        if count >= num_examples:
            break
        
        # allow missing/invalid entries to be skipped
        if not isinstance(example, dict):
            continue

        video_index = example.get('video_index', '')
        vlm_summary = example.get('vlm_summary', '')
        structured_details = example.get('structured_details', '')
        original_questions = example.get('questions', '')
        
        print(f"Processing example {count + 1}/{num_examples} (video_index: {video_index})...")
        
        generated_questions = qwen_model.generate_questions(vlm_summary, structured_details)
        
        video_file = f"video{video_index}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_file)
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}. Skipping this example.")
            continue

        print("\n Getting VLM answers to generated questions...")
        try:
            vlm_answers = gemini.answer_questions(video_path, generated_questions)
        except Exception as e:
            print(f"Warning: Failed to get VLM answers for video{video_index}: {e}")
            vlm_answers = ""
            qa_caption = ""
            continue

        print("→ Creating QA captions...")
        try:
            qa_caption = gemini.generate_qa_caption(vlm_summary, vlm_answers)
        except Exception as e:
            print(f"Warning: Failed to generate QA caption for video{video_index}: {e}")
            qa_caption = ""
            continue

        try:
            captions_filename = f"video{video_index}_results.txt"
            captions_path = os.path.join(captions_dir, captions_filename)
            with open(captions_path, 'w', encoding='utf-8') as cf:
                cf.write("=== VLM SUMMARY ===\n")
                cf.write((vlm_summary or "").strip() + "\n\n")
                cf.write("=== QA CAPTION ===\n")
                cf.write((qa_caption or "").strip() + "\n\n")
                cf.write("=== NON-QA CAPTION ===\n")
                cf.write("\n")
        except Exception as e:
            print(f"Warning: Failed to write caption file for video{video_index}: {e}")

        output_entry = {
            'video_index': video_index,
            'vlm_summary': vlm_summary,
            'structured_details': structured_details,
            'original_questions': original_questions,
            'generated_questions': generated_questions,
            'vlm_answers': vlm_answers,
            'qa_caption': qa_caption
        }
        
        results.append(output_entry)
        count += 1
    
    # Write results to output JSONL file
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nProcessed {len(results)} examples.")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON file (array of examples) with SFT pipeline")
    parser.add_argument('--input', type=str, default='evaluation_NQA_results.json', help='Input JSON file path')
    parser.add_argument('--output', type=str, default='output_sft.jsonl', help='Output JSONL file path')
    parser.add_argument('--num_examples', type=int, default=100, help='Number of examples to process')
    args = parser.parse_args()
    
    # Verify input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found.")
        exit(1)
    
    process_json_file(args.input, args.output, args.num_examples)