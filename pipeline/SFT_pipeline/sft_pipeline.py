import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"  
ADAPTER_DIR = "./qwen3-30b-instruct-police-questions-lora-gemini-vlm"


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_DIR, torch_dtype=torch.float16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def generate_response(model, tokenizer, vlm_summary, structured_details):
    """Generate investigative questions for a single example."""
    prompt = f"""
        You are an AI assistant aiding law enforcement analysts reviewing body-worn camera footage.

        Your task:
        - Based on the provided structured details, generate a list of investigative questions.
        - Every question must be something a human could answer by watching the video.
        - The goal is to guide analysts toward visual clues, context, behavior, or environment details that may matter.

        Rules for your output:
        - Write a total of 4 meaningful questions that can extract the most visual information.
        - Do NOT repeat facts already stated.
        - Focus areas include: body language, environment, timeline, objects, threat indicators, interaction dynamics, or visual anomalies.
        - Use clear, concise, professional language.
        - Format the output as a numbered list.

        Structured information provided:
        {structured_details}
        """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def process_jsonl_file(input_file, output_file, num_examples=5):
    """Process a JSONL file and generate outputs."""
    model, tokenizer = load_model_and_tokenizer()
    
    results = []
    count = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            if count >= num_examples:
                break
            
            try:
                example = json.loads(line.strip())
                
                video_index = example.get('video_index', '')
                vlm_summary = example.get('vlm_summary', '')
                structured_details = example.get('structured_details', '')
                original_questions = example.get('questions', '')
                
                print(f"Processing example {count + 1}/{num_examples} (video_index: {video_index})...")
                
                # Generate questions using the model
                generated_questions = generate_response(model, tokenizer, vlm_summary, structured_details)
                
                # Create output entry
                output_entry = {
                    'video_index': video_index,
                    'vlm_summary': vlm_summary,
                    'structured_details': structured_details,
                    'original_questions': original_questions,
                    'generated_questions': generated_questions,
                }
                
                results.append(output_entry)
                count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line {count + 1}: {e}")
                continue
    
    # Write results to output JSONL file
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nProcessed {len(results)} examples.")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL file with SFT pipeline")
    parser.add_argument('--input', type=str, default='./distillation_results_gemini.jsonl', help='Input JSONL file path')
    parser.add_argument('--output', type=str, default='output_sft.jsonl', help='Output JSONL file path')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of examples to process')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found.")
        exit(1)
    
    process_jsonl_file(args.input, args.output, args.num_examples)