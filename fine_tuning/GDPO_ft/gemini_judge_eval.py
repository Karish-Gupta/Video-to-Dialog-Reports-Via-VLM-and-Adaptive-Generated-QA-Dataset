import re
import os
import json
import argparse
import time
import concurrent.futures
from tqdm import tqdm
from unsloth import FastLanguageModel
from vllm import LLM, SamplingParams
from datasets import load_dataset
from fine_tuning.GDPO_ft.utils import apply_prompt_template, JUDGE_PROMPT_TEMPLATE
from fine_tuning.GDPO_ft.rewards import call_api
from models.gemini_model import gemini_model


def load_trained_model(model_path: str, max_seq_length: int = 4096):

    print(f"Loading trained model from {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
        gpu_memory_utilization=0.90,
    )
    
    # Prepare model for inference
    model = FastLanguageModel.for_inference(model)
    
    print("Model loaded successfully")
    return model, tokenizer


def generate_questions_batch(model, tokenizer, structured_details_list, 
                             max_tokens: int = 2048, temperature: float = 0.7):
    
    prompts = []
    for details in structured_details_list:
        example = {"structured_details": details}
        prompt_dict = apply_prompt_template(example, tokenizer)
        prompts.append(prompt_dict["prompt"])
    
    # Use text_generation pipeline for inference
    from transformers import pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    
    outputs = pipe(prompts, max_new_tokens=max_tokens, temperature=temperature, 
                   do_sample=True, return_full_text=False)
    
    generated_questions = []
    for output in outputs:
        text = output[0]["generated_text"].strip() if isinstance(output, list) else output
        # Extract questions from the output
        match = re.search(r"<question>(.*?)</question>", text, re.DOTALL)
        if match:
            generated_questions.append(match.group(1).strip())
        else:
            generated_questions.append(text)
    
    return generated_questions


def judge_questions_with_gemini(generated_questions: list, gold_questions_list: list, 
                                 max_workers: int = 4) -> dict:
    
    gemini = gemini_model()
    prompts_map = {}
    
    # Prepare prompts
    for i, (gen_qs, gold_qs) in enumerate(zip(generated_questions, gold_questions_list)):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            gold_questions=gold_qs,
            questions=gen_qs
        )
        prompts_map[i] = prompt
    
    if not prompts_map:
        print("No valid questions to judge")
        return {i: 0.0 for i in range(len(generated_questions))}
    
    # Parallel Execution
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, prompt in prompts_map.items():
            time.sleep(0.1)  # Small delay to avoid rate limits
            future_to_idx[executor.submit(call_api, idx, prompt)] = idx
        
        # Collect results as they finish
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), 
                          total=len(future_to_idx), desc="Judging questions"):
            idx, response_text = future.result()
            results[idx] = response_text
    
    # Parse Results
    final_scores = {}
    for idx, response in results.items():
        if response is None:
            final_scores[idx] = 0.0
            continue
        
        clean_response = response.strip()
        
        # Parse "1 0 1 1" pattern
        matches = re.search(r"([01])\D*([01])\D*([01])\D*([01])", clean_response)
        if matches:
            scores = [int(matches.group(k)) for k in range(1, 5)]
            final_scores[idx] = sum(scores) / 4.0
        else:
            final_scores[idx] = 0.0
    
    return final_scores


def evaluate_on_dataset(model_path: str, dataset_path: str, output_file: str = "eval_results.jsonl"):
    
    # Load model
    model, tokenizer = load_trained_model(model_path)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Process in batches
    batch_size = 4
    all_results = []
    
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]
        
        structured_details_list = batch["structured_details"]
        gold_questions_list = batch.get("questions", [""] * len(batch))
        
        # Generate questions
        generated_questions = generate_questions_batch(model, tokenizer, structured_details_list)
        
        # Judge with Gemini
        scores = judge_questions_with_gemini(generated_questions, gold_questions_list)
        
        # Store results
        for i, (details, gen_qs, gold_qs) in enumerate(zip(structured_details_list, 
                                                             generated_questions, 
                                                             gold_questions_list)):
            result = {
                "structured_details": details,
                "generated_questions": gen_qs,
                "gold_questions": gold_qs,
                "gemini_score": scores.get(i, 0.0)
            }
            all_results.append(result)
    
    # Save results
    print(f"Saving evaluation results to {output_file}...")
    with open(output_file, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    
    # Print summary statistics
    scores_list = [r["gemini_score"] for r in all_results]
    print("\n" + "="*50)
    print("Evaluation sumary:")
    print("="*50)
    print(f"Total samples: {len(all_results)}")
    print(f"Average Gemini score: {sum(scores_list) / len(scores_list):.4f}")
    print(f"Min score: {min(scores_list):.4f}")
    print(f"Max score: {max(scores_list):.4f}")
    print(f"Results saved to: {output_file}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained RL model with Gemini judge")
    parser.add_argument("--model_path", type=str, default="grpo_saved_model",
                       help="Path to the trained model")
    parser.add_argument("--dataset_path", type=str, default="fine_tuning/GDPO_ft/rl_training_data.jsonl",
                       help="Path to the evaluation dataset")
    parser.add_argument("--output_file", type=str, default="eval_results.jsonl",
                       help="Output file for results")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    evaluate_on_dataset(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_file=args.output_file
    )
