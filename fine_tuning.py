import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
from tqdm import tqdm

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])


class FineTuner(Dataset):
    
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                
                # Use the raw JSON object directly
                if "structured_details" in obj and "questions" in obj:
                    self.examples.append(obj)
        
        print(f"Loaded {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input and output separately
        prompt = f"Input: {example['structured_details']}\nOutput: "
        full_text = f"Input: {example['structured_details']}\nOutput: {example['questions']}"
        
        # Tokenize prompt (input part)
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        # Tokenize full text (input + output)
        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels: -100 for input tokens (ignored in loss), actual tokens for output
        labels = full_tokens["input_ids"].clone().squeeze()
        prompt_length = len(prompt_tokens["input_ids"])
        labels[:prompt_length] = -100  # Mask input tokens
        
        return {
            "input_ids": full_tokens["input_ids"].squeeze(),
            "attention_mask": full_tokens["attention_mask"].squeeze(),
            "labels": labels
        }


def train(
    data_path="distillation_results.jsonl",
    model_name="meta-llama/Meta-Llama-3-8B",
    output_dir="./fine_tuned_model",
    epochs=3,
    batch_size=2,
    learning_rate=2e-4,
    max_length=512,
    gradient_accumulation_steps=4
):
    """Simple fine-tuning loop"""
    
    # Load tokenizer
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    print("Loading model")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )
    
    # Add LoRA adapter
    print("Adding LoRA adapter")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create dataset and dataloader
    print("Loading dataset")
    dataset = FineTuner(data_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"Starting training for {epochs} epochs")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Track loss
            running_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
        
        epoch_loss = running_loss / len(dataloader)
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # Save model
    print(f"Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete")


if __name__ == "__main__":
    train(
        data_path="distillation_results.jsonl",
        model_name="meta-llama/Meta-Llama-3-8B",
        output_dir="./fine_tuned_model",
        epochs=3,
        batch_size=2,
        learning_rate=2e-4,
        max_length=512,
        gradient_accumulation_steps=4
    )


