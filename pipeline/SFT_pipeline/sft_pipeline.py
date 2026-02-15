import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
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
    stop_marker = "<END_OF_QUESTIONS>"

    prompt = f"""You are an AI assistant aiding law enforcement analysts reviewing body-worn camera footage.

Your task: Based on the provided structured details, generate exactly 4 investigative questions.

Rules for your output:
- Write exactly 4 numbered questions (1.-4.) formatted as a numbered list.
- Do NOT repeat facts already stated.
- Use clear, concise, professional language.

Structured information provided:
{structured_details}

End the output by placing the following stop marker on its own line after question 4:
{stop_marker}

Generated questions:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    num_input_tokens = inputs['input_ids'].shape[1]

    stop_ids = tokenizer.encode(stop_marker, add_special_tokens=False)

    class StopOnSequence(StoppingCriteria):
        def __init__(self, stop_ids):
            self.stop_ids = stop_ids
        def __call__(self, input_ids, scores, **kwargs):
            # only check the last len(stop_ids) tokens
            seq = input_ids[0].tolist()
            n = len(self.stop_ids)
            if n == 0 or len(seq) < n:
                return False
            return seq[-n:] == self.stop_ids

    stopping_criteria = StoppingCriteriaList([StopOnSequence(stop_ids)])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
        )

    gen_tokens = outputs[0][num_input_tokens:]
    generated_raw = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    if stop_marker in generated_raw:
        generated = generated_raw.split(stop_marker)[0].strip()
    else:
        generated = generated_raw.strip()

    import re
    m = re.search(r"(1\.[\s\S]*?4\.[\s\S]*?)(?=\n\s*\d+\.|\Z)", generated)
    if m:
        extracted = m.group(1).strip()
        return extracted

    lines = [l for l in generated.splitlines() if re.match(r"^\s*\d+\.", l)]
    if lines:
        numbered = []
        for l in lines:
            if len(numbered) >= 4:
                break
            numbered.append(l.strip())
        return "\n".join(numbered)

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