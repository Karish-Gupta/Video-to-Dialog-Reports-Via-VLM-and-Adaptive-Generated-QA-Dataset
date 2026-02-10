from unsloth import FastLanguageModel
import random
import torch
from datasets import load_dataset
from fine_tuning.GDPO_ft.utils import *

dataset_name = "fine_tuning/GDPO_ft/rl_training_data.jsonl"
model_path = "grpo_saved_model"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Load and preprocess dataset
dataset = load_dataset("json", data_files=dataset_name, split="train")
dataset = dataset.map(lambda x: apply_prompt_template(x, tokenizer))

indices = random.sample(range(len(dataset)), 3)

print("\n" + "="*50)
print(f"Starting Inference on 3 Random Examples")
print("="*50 + "\n")

# Inference
model.eval() 

for i, idx in enumerate(indices):
    example = dataset[idx]
    input_prompt = example.get('prompt') 
    gold_answer = example.get('questions') 

    print(f"--- Example {i+1} (Index {idx}) ---")

    # Move inputs to GPU correctly
    inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024,
            temperature=0.1,    # Low temp is good for evaluation
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    
    generated_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Print Side-by-Side
    print(f"\n[GOLD ANSWER]:\n{gold_answer}\n")
    print(f"[MODEL GENERATED]:\n{generated_response}\n")
    print("-" * 50 + "\n")