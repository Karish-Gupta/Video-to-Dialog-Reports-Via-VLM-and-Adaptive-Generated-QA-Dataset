import random
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from fine_tuning.GDPO_ft.utils import *

# Point to your FINAL saved folder
adapter_path = "grpo_saved_model" 
base_model_name = "Qwen/Qwen3-4B-Thinking-2507"
dataset_name = "fine_tuning/GDPO_ft/rl_training_data.jsonl"

# Quantization config for base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # Half precision
    bnb_4bit_quant_type="nf4",            # Normal Float 4
    bnb_4bit_use_double_quant=True,
)
# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
    )

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Load model with adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# Load and preprocess dataset
dataset = load_dataset("json", data_files=dataset_name, split="train")
dataset = dataset.map(lambda x: apply_prompt_template(x, tokenizer))

indices = random.sample(range(len(dataset)), 3)

print("\n" + "="*50)
print(f"Starting Inference on 3 Random Examples")
print("="*50 + "\n")

# Inference
for i, idx in enumerate(indices):
    example = dataset[idx]
    
    input_prompt = example.get('prompt') 
    
    gold_answer = example.get('questions')

    print(f"--- Example {i+1} (Index {idx}) ---")
    


    inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.1,    # Deterministic 
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the output
    generated_response = generated_text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    
    # Print Side-by-Side
    print(f"\n[GOLD ANSWER]:\n{gold_answer}\n")
    print(f"[MODEL GENERATED]:\n{generated_response}\n")
    print("-" * 50 + "\n")