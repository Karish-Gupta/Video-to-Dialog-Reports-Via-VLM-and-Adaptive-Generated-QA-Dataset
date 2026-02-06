import os
from huggingface_hub import login
from fine_tuning.distillation_ft.distillation_ft import distillation_ft

if __name__ == "__main__":
   
   # Login
   if "HF_TOKEN" in os.environ:
      login(token=os.environ["HF_TOKEN"])
   
   # Configs
   dataset_path = "fine_tuning/distillation_results_gemini.jsonl"
   model_name = "Qwen/Qwen3-72B-Instruct"

   # Initialize Runner
   ft_runner = distillation_ft(
      model_name=model_name,
      training_dataset=dataset_path,  
      testing_dataset=None,     
      train_batch_size=2,
      eval_batch_size=2,
      gradient_accumulation_steps=4,
      num_epochs=15,
      learning_rate=2e-4,
      max_input_length=1024,
      max_target_length=512,
   )

   ft_runner.preprocess_dataset()
   ft_runner.init_model()
   ft_runner.train()
   
   
   # Cleanup
   ft_runner.cleanup()
   del ft_runner # Explicit delete to free pointer
   
