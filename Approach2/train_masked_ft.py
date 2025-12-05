"""
Fine-tuning script for Approach2:
- Trains model to generate questions from MASKED structured details
- Adapted from fine_tuning/distillation_ft.py
"""

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from Approach2.model_utils.eval_utils import *
from Approach2.model_utils.preprocessing import * 
import os


# Login to HF CLI
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])


class MaskedDistillationFT:
    """
    Fine-tuning class for Approach2: Training with masked structured details.
    """
    def __init__(
        self,
        model_name,
        dataset_name,  # Path to masked dataset
        train_batch_size,
        eval_batch_size,
        gradient_accumulation_steps,
        num_epochs,
        learning_rate,
        max_input_length,
        max_target_length,
        lora_r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        lora_target_modules=None,  # if None, good defaults for LLaMA
        lora_bias="none",          # "none" | "lora_only" | "all"
    ):
        # Configs
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LoRA configs
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.lora_bias = lora_bias

        # Setup
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None

    def preprocess_dataset(self, train_size, eval_size, seed=101):
        """Initialize tokenizer and preprocess masked dataset."""
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Llama 3 specific padding fix
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Call the preprocessing function for masked data
        self.train_loader, self.val_loader = preprocess_masked_dataset(
            tokenizer=self.tokenizer,
            dataset_name=self.dataset_name,
            train_size=train_size,
            eval_size=eval_size,
            max_input_length=self.max_input_length,
            max_target_length=self.max_target_length,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            seed=seed
        )

    def init_model(self):
        """Initialize model with LoRA configuration."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="auto",  # Changed to auto for better multi-gpu or generic support
            torch_dtype=torch.float16
        )
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()

        # Llama 3 target modules
        if self.lora_target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = self.lora_target_modules

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias=self.lora_bias,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train(self):
        """Training loop."""
        self.model.train()

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        total_steps = len(self.train_loader) * self.num_epochs
        print(f"Total training steps: {total_steps}")
        print(f"Effective batch size: {self.train_batch_size * self.gradient_accumulation_steps}")

        global_step = 0
        for epoch in range(self.num_epochs):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch + 1}/{self.num_epochs}")
            print(f"{'='*80}")

            epoch_loss = 0.0
            self.optimizer.zero_grad()

            for step, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item() * self.gradient_accumulation_steps

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % 10 == 0:
                        avg_loss = epoch_loss / (step + 1)
                        print(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f}")

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"\nEpoch {epoch + 1} completed | Avg Loss: {avg_epoch_loss:.4f}")

            # Evaluate after each epoch
            print(f"\nRunning evaluation for Epoch {epoch + 1}...")
            eval_results = evaluate_model(
                model=self.model,
                val_loader=self.val_loader,
                device=self.device,
                tokenizer=self.tokenizer,
                max_gen_length=256,
                show_samples=3
            )

            print(f"\nEpoch {epoch + 1} Evaluation Results:")
            print(f"  Exact Match: {eval_results['exact_match_accuracy']:.4f}")
            print(f"  F1 Score: {eval_results['f1']:.4f}")

    def save_model(self, output_dir):
        """Save the fine-tuned model."""
        print(f"\nSaving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Model saved successfully!")

    def run_full_pipeline(self, train_size, eval_size, output_dir, seed=101):
        """Run the complete training pipeline."""
        print("\n" + "="*80)
        print("APPROACH 2: MASKED FINE-TUNING PIPELINE")
        print("="*80)
        
        print("\n[1/4] Preprocessing dataset...")
        self.preprocess_dataset(train_size, eval_size, seed)
        
        print("\n[2/4] Initializing model...")
        self.init_model()
        
        print("\n[3/4] Training...")
        self.train()
        
        print("\n[4/4] Saving model...")
        self.save_model(output_dir)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)


if __name__ == "__main__":
    
    trainer = MaskedDistillationFT(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        dataset_name="Approach2/masked_data.jsonl",
        train_batch_size=1,
        eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_epochs=3,
        learning_rate=2e-5,
        max_input_length=1024,
        max_target_length=512,
        lora_r=64,
        lora_alpha=64,
        lora_dropout=0.05
    )
    
    # Run training pipeline
    trainer.run_full_pipeline(
        train_size=100,
        eval_size=20,
        output_dir="approach2_finetuned_model",
        seed=101
    )
