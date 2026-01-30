import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from rewards import *

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure Training
# Note: You can tune the weights of your rewards here
# GDPO Specific: Ensure you are using the fork, or these will just run as standard GRPO
training_args = GRPOConfig(
    output_dir="gdpo_output",
    learning_rate=1e-6,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_prompt_length=256,
    max_completion_length=256,
    num_generations=4,      # Number of samples per prompt (Group size)
    logging_steps=10,
)

# 4. Initialize Trainer with Multiple Rewards
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[accuracy_reward, format_reward, len_penalty_reward], # Pass list of funcs
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()