import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fine_tuning.GDPO_ft.rewards import format_complexity_reward
from fine_tuning.GDPO_ft.llm_judge import judge_reward
from fine_tuning.GDPO_ft.utils import apply_prompt_template

# Model and dataset paths
model_name = "Qwen/Qwen3-4B-Thinking-2507"
dataset_name = "rl_train_data"

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, # Half precision
    bnb_4bit_quant_type="nf4",             # Normal Float 4
    bnb_4bit_use_double_quant=True,
)
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda:0"
    )

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load and preprocess dataset
dataset = load_dataset(dataset_name)
dataset = dataset.map(lambda x: apply_prompt_template(x, tokenizer))

# For GDPO ensure you are using the fork, or these will just run as standard GRPO
training_args = GRPOConfig(
    output_dir="grpo_output",
    learning_rate=1e-6,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=1024,
    num_generations=4,      # Number of samples per prompt (Group size)
    logging_steps=10,
)

# Initialize Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_complexity_reward, judge_reward],
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()