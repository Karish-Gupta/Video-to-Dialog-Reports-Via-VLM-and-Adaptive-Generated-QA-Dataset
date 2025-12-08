import torch
import json
import os
import numpy as np
import optuna
from sklearn.model_selection import KFold
from huggingface_hub import login
from fine_tuning.distillation_ft.distillation_ft import distillation_ft
from fine_tuning.model_utils.helpers import load_jsonl, save_jsonl


def objective(trial):
   # Define Hyperparameter Search Space
   learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
   lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
   lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64, 128])
   lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)
   batchsize = trial.suggest_categorical("batch_size", [1, 2])

   # Constants
   TUNING_EPOCHS = 10
   dataset_path = "fine_tuning/distillation_results_full.jsonl" 
   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
   N_FOLDS = 5

   # Load Data & Prepare K-Fold
   full_dataset = load_jsonl(dataset_path)
   kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
   
   fold_scores = []

   print(f"\n--- Starting Trial {trial.number} with {N_FOLDS}-Fold CV ---")

   # K-Fold Loop
   for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
      
      # Create Split
      train_data = [full_dataset[i] for i in train_idx]
      val_data = [full_dataset[i] for i in val_idx]
      
      # Create Unique Temp Files (Use trial number to prevent collisions)
      temp_train = f"temp_train_t{trial.number}_f{fold}.jsonl"
      temp_val = f"temp_val_t{trial.number}_f{fold}.jsonl"
      save_jsonl(train_data, temp_train)
      save_jsonl(val_data, temp_val)

      # Initialize Runner
      ft_runner = distillation_ft(
         model_name=model_name,
         training_dataset=temp_train, 
         testing_dataset=temp_val,
         train_batch_size=batchsize, 
         eval_batch_size=batchsize,
         gradient_accumulation_steps=4, # Safe value for small batches
         num_epochs=TUNING_EPOCHS,
         learning_rate=learning_rate,
         max_input_length=1024,
         max_target_length=512,
         lora_r=lora_r,
         lora_alpha=lora_alpha,
         lora_dropout=lora_dropout,
      )

      try:
         # Execution
         ft_runner.preprocess_dataset()
         ft_runner.init_model()     
         ft_runner.train()
         
         # Evaluate
         f1, bertscore = ft_runner.evaluate()
         combined_score = (f1 + bertscore) / 2
         fold_scores.append(combined_score)
         
         # Report fold progress
         print(f" Trial {trial.number} | Fold {fold+1}: {combined_score:.4f}")

         # Report the average SO FAR to Optuna to prune bad trials early
         current_avg = np.mean(fold_scores)
         trial.report(current_avg, fold)
         
         if trial.should_prune():
               raise optuna.exceptions.TrialPruned()

      except RuntimeError as e:
         # Handle OOMs by pruning the trial gracefully
         if "CUDA out of memory" in str(e):
               print(f"Trial {trial.number} failed due to OOM.")
               raise optuna.exceptions.TrialPruned()
         else:
               raise e
               
      finally:
         # Cleanup resources after EACH fold
         ft_runner.cleanup()
         del ft_runner
         torch.cuda.empty_cache()
         
         # Remove temp files
         if os.path.exists(temp_train): os.remove(temp_train)
         if os.path.exists(temp_val): os.remove(temp_val)

   # 4. Return Average Score
   final_avg_score = np.mean(fold_scores)
   return final_avg_score

if __name__ == "__main__":
   # Login
   if "HF_TOKEN" in os.environ:
      login(token=os.environ["HF_TOKEN"])

   # Create Study
   study = optuna.create_study(
      direction="maximize", 
      pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
   )
   
   print("Starting Hyperparameter Tuning...")
   # Run optimization
   study.optimize(objective, n_trials=15) 

   print("\n" + "="*60)
   print("Hyperparameter Tuning Complete")
   print(f"Best Trial Value (Avg Combined Score): {study.best_value:.4f}")
   print("Best Parameters:")
   for key, value in study.best_params.items():
      print(f"  {key}: {value}")
   print("="*60 + "\n")
   
   # Save best params
   with open("best_hyperparameters.json", "w") as f:
      json.dump(study.best_params, f, indent=4)