from huggingface_hub import login
import os
from fine_tuning.distillation_ft.distillation_ft import distillation_ft

if __name__ == "__main__":
    
   # Login to HF CLI
   if "HF_TOKEN" in os.environ:
      login(token=os.environ["HF_TOKEN"])
   
   # Model Configs
   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
   
   # Update this to your local JSONL file
   dataset_name = "fine_tuning/distillation_results_full.jsonl" 
   
   # Training Configs
   train_batch_size = 4
   eval_batch_size = 4
   gradient_accumulation_steps = 8 # Adjusted: 32 might be too slow for small datasets, 8 is usually stable
   num_epochs = 15
   learning_rate = 2e-4
   
   # Lengths
   max_input_length = 1024 # Increased for structured details
   max_target_length = 512 # Enough for 4 questions
   
   # Dataset Sizes (Adjust based on your actual file size)
   train_size = 24 
   eval_size = 6

   ft_runner = distillation_ft(
      model_name=model_name,
      dataset_name=dataset_name,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      gradient_accumulation_steps=gradient_accumulation_steps,
      num_epochs=num_epochs,
      learning_rate=learning_rate,
      max_input_length=max_input_length,
      max_target_length=max_target_length,
   )

   ft_runner.preprocess_dataset(train_size=train_size, eval_size=eval_size, seed=101)
   ft_runner.init_model()
   
   print(f"Starting Fine-tuning for {model_name}")
   print(f"Dataset: {dataset_name}")
   
   ft_runner.train()
   ft_runner.evaluate()
   ft_runner.cleanup()