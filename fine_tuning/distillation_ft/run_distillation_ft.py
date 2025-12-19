import json
import os
import numpy as np
import torch
from sklearn.model_selection import KFold
from huggingface_hub import login
from fine_tuning.distillation_ft.distillation_ft import distillation_ft
from fine_tuning.model_utils.helpers import load_jsonl, save_jsonl

if __name__ == "__main__":
   
   # Login
   if "HF_TOKEN" in os.environ:
      login(token=os.environ["HF_TOKEN"])
   
   # Configs
   original_dataset_path = "fine_tuning/distillation_results_gemini.jsonl"
   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
   
   N_FOLDS = 5
   results = {
      "f1": [],
      "bertscore": [],
   }

   # Load Full Data
   full_dataset = load_jsonl(original_dataset_path)
   print(f"Loaded {len(full_dataset)} examples from {original_dataset_path}")

   # Initialize K-Fold (5 Splits = 24 Train / 6 Val)
   kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

   print(f"\n{'='*40}")
   print(f"Starting {N_FOLDS}-Fold Cross Validation")
   print(f"{'='*40}\n")

   # The K-Fold Loop
   for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
      print(f"--- Running Fold {fold + 1}/{N_FOLDS} ---")
      
      # Create Fold Files
      train_data = [full_dataset[i] for i in train_idx]
      val_data = [full_dataset[i] for i in val_idx]
      
      # Save temp files for this specific fold
      temp_train_file = f"temp_train_fold_{fold}.jsonl"
      temp_val_file = f"temp_val_fold_{fold}.jsonl"
      save_jsonl(train_data, temp_train_file)
      save_jsonl(val_data, temp_val_file)

      # Initialize Runner
      ft_runner = distillation_ft(
         model_name=model_name,
         training_dataset=temp_train_file,  
         testing_dataset=temp_val_file,     
         train_batch_size=2,
         eval_batch_size=2,
         gradient_accumulation_steps=4,
         num_epochs=15,
         learning_rate=2e-4,
         max_input_length=1024,
         max_target_length=512,
      )

      try:
         ft_runner.preprocess_dataset()
         ft_runner.init_model()
         ft_runner.train()
         ft_runner.train()
         
         # Store results
         f1, bert = ft_runner.evaluate()
         results["f1"].append(f1)
         results["bertscore"].append(bert)
         
         print(f"Fold {fold+1} Result - F1: {f1:.4f}, BERT: {bert:.4f}")

      except Exception as e:
         print(f"Error in Fold {fold + 1}: {e}")
         
      finally:
         # Cleanup
         ft_runner.cleanup()
         del ft_runner # Explicit delete to free pointer
         
         # Remove temp files
         if os.path.exists(temp_train_file): os.remove(temp_train_file)
         if os.path.exists(temp_val_file): os.remove(temp_val_file)

   # Print final summary
   print(f"\n{'='*40}")
   print("Cross Validation Complete")
   if results["f1"]:
       print(f"Average F1: {np.mean(results['f1']):.4f} (+/- {np.std(results['f1']):.4f})")
       print(f"Average BERTScore: {np.mean(results['bertscore']):.4f}")
   else:
       print("No results collected.")
   print(f"{'='*40}\n")