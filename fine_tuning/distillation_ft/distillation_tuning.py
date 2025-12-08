import torch
from huggingface_hub import login
import optuna
from fine_tuning.distillation_ft.distillation_ft import distillation_ft


def objective(trial):
   # 1. Define Hyperparameter Search Space
   learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
   lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
   lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64, 128])
   lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)
   batchsize = trial.suggest_categorical("batch_size", [1, 2])

   TUNING_EPOCHS = 10
   
   # Setup Runner
   dataset_name = "fine_tuning/distillation_results_full.jsonl" 
   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

   ft_runner = distillation_ft(
      model_name=model_name,
      dataset_name=dataset_name,
      train_batch_size=batchsize, 
      eval_batch_size=batchsize,
      gradient_accumulation_steps=8,
      num_epochs=TUNING_EPOCHS,
      learning_rate=learning_rate,
      max_input_length=1028,
      max_target_length=512,
      lora_r=lora_r,
      lora_alpha=lora_alpha,
      lora_dropout=lora_dropout,
   )

   try:
      # Execution
      ft_runner.preprocess_dataset(train_size=26, eval_size=6, seed=101)
      ft_runner.init_model()     
      ft_runner.train()
      f1, bertscore = ft_runner.evaluate()
      combined_score = (f1 + bertscore) / 2
      return combined_score

   except RuntimeError as e:
      raise e
         
   finally:
      # Cleanup to prevent OOM in next trial
      ft_runner.cleanup()

if __name__ == "__main__":
   # Create Study
   study = optuna.create_study(
      direction="maximize", 
      pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
   )
   
   print("Starting Hyperparameter Tuning...")
   # Run optimization
   study.optimize(objective, n_trials=15) # Set n_trials to desired number

   print("\n" + "="*60)
   print("Hyperparameter Tuning Complete")
   print(f"Best Trial Value (Val Loss): {study.best_value:.4f}")
   print("Best Parameters:")
   for key, value in study.best_params.items():
      print(f"  {key}: {value}")
   print("="*60 + "\n")
   
   # Save best params to file
   import json
   with open("best_hyperparameters.json", "w") as f:
      json.dump(study.best_params, f, indent=4)