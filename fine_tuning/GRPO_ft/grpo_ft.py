from unsloth import FastLanguageModel, PatchFastRL
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from fine_tuning.GRPO_ft.rewards import format_complexity_reward, gemini_judge_reward
from fine_tuning.GRPO_ft.utils import apply_prompt_template

PatchFastRL("GRPO", FastLanguageModel) # Required to patch TRL for Unsloth

# Configuration
model_name = "Qwen/Qwen3-4B-Thinking-2507"
dataset_name = "fine_tuning/GRPO_ft/rl_training_data.jsonl"
max_seq_length = 4096 # Unsloth supports long context easily

# Unsloth handles 4-bit loading internally
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None, # Auto-detects (bfloat16 for H100)
    load_in_4bit = True,
    gpu_memory_utilization = 0.90, # Use 90% of GPU memory
)

# Apply LoRA directly to the model
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 101,
)

tokenizer.padding_side = "left" 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id # Ensure model and tokenizer are aligned on padding token

print("Model and tokenizer loaded successfully")

# Load and preprocess dataset
dataset = load_dataset("json", data_files=dataset_name, split="train")
dataset = dataset.map(lambda x: apply_prompt_template(x, tokenizer))
print("Dataset loaded and preprocessed successfully")

# For GDPO ensure you are using the fork, or these will just run as standard GRPO
training_args = GRPOConfig(
    output_dir="grpo_output",
    learning_rate=5e-6,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_prompt_length=1024,
    max_completion_length=2048,
    num_generations=8,
    logging_steps=10,             
    gradient_checkpointing=True,
    bf16=True,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2
)

# Initialize Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_complexity_reward, gemini_judge_reward],
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
    )

trainer.train()

# Save fine-tuned model and tokenizer
model.save_pretrained("grpo_saved_model")
tokenizer.save_pretrained("grpo_saved_model")
print("Model and tokenizer saved to grpo_saved_model/")