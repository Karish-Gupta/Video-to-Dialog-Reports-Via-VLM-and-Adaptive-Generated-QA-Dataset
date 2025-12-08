import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator

def collate_eval(batch):
   """
   Collate function that keeps the target_text as a list for metric computation,
   while batching input_ids/attention_mask for the model.
   """
   input_ids = torch.stack([torch.tensor(b["input_ids"]) for b in batch])
   attention_mask = torch.stack([torch.tensor(b["attention_mask"]) for b in batch])
   
   # Keep target_text raw for pure text comparison (BLEU/ROUGE/BERTScore)
   target_texts = [b["target_text"] for b in batch]

   return {
      "input_ids": input_ids,
      "attention_mask": attention_mask,
      "target_text": target_texts,
   }

def preprocess_dataset(
   tokenizer,
   training_dataset, 
   testing_dataset,
   max_input_length,
   max_target_length,
   train_batch_size,
   eval_batch_size,
):
   """
   Preprocess dataset for Llama 3 instruction fine-tuning.
   Returns train_loader, val_loader.
   """
   
   # Load Datasets from the file paths 
   raw_train_dataset = load_dataset("json", data_files=training_dataset, split="train") # Use split="train" because load_dataset defaults to a 'train' key for json files
   
   # Handle case where testing_dataset might be None or empty (if running full training without eval)
   if testing_dataset:
      raw_eval_dataset = load_dataset("json", data_files=testing_dataset, split="train")
   else:
      raw_eval_dataset = None

   # Define System Prompt
   system_prompt = """
   You are an AI assistant aiding law enforcement analysts reviewing body-worn camera footage.

   Your task:
   - Based on the provided structured details, generate a list of investigative questions.
   - Every question must be something a human could answer by watching the video.
   - The goal is to guide analysts toward visual clues, context, behavior, or environment details that may matter.

   Rules for your output:
   - Write a total of 4 meaningful questions that can extract the most visual information.
   - Each question should pertain to one of the four categories (scene-level, entity-level, action-level, semantic-level).
   - Do NOT repeat facts already stated.
   - Focus areas include: body language, environment, timeline, objects, threat indicators, interaction dynamics, or visual anomalies.
   - Use clear, concise, professional language.
   - Format the output as a numbered list.
   """

   # Tokenize Function for Training
   def tokenize_train(example):
      # Construct the conversation for Llama 3
      messages = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": f"Structured information provided:\n {example['structured_details']}"}
      ]

      # Apply chat template for the Input (Prompt)
      input_text = tokenizer.apply_chat_template(
         messages,
         add_generation_prompt=True, 
         tokenize=False
      )

      # Construct the full text (Prompt + Answer)
      target_text = example['questions']
      messages_with_answer = messages + [{"role": "assistant", "content": target_text}]
      full_text = tokenizer.apply_chat_template(messages_with_answer, tokenize=False)

      # Tokenize Input only (to calculate length for masking)
      input_tokenized = tokenizer(
         input_text, 
         add_special_tokens=False 
      )

      # Tokenize Full Text
      tokenized = tokenizer(
         full_text,
         max_length=max_input_length + max_target_length, 
         truncation=True, 
         padding="max_length", 
         add_special_tokens=False 
      )

      # Create Labels
      # Mask the input part with -100 so we don't train on the prompt, only the answer
      input_length = len(input_tokenized["input_ids"])
      labels = tokenized["input_ids"].copy()
      
      # Set all prompt tokens to -100
      labels[:input_length] = [-100] * input_length
      
      # Mask padding tokens
      labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

      tokenized["labels"] = labels
      return tokenized

   # Tokenize Function for Evaluation
   def tokenize_eval(example):
      # Match the Training input structure
      messages = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": f"Structured information provided:\n {example['structured_details']}"}
      ]

      # Get input text formatted with chat template
      input_text = tokenizer.apply_chat_template(
         messages, 
         add_generation_prompt=True, 
         tokenize=False
      )

      # Tokenize input only
      tokenized = tokenizer(
         input_text, 
         max_length=max_input_length, 
         truncation=True, 
         padding="max_length", 
         add_special_tokens=False
      )

      # Store the target answer raw text for evaluation metrics
      tokenized["target_text"] = example['questions']

      return tokenized

   # Apply Mapping
   print(f"Tokenizing training data from {training_dataset}...")
   train_dataset = raw_train_dataset.map(
      tokenize_train, 
      batched=False, 
      remove_columns=raw_train_dataset.column_names
   )
   
   if raw_eval_dataset:
      print(f"Tokenizing validation data from {testing_dataset}...")
      eval_dataset = raw_eval_dataset.map(
         tokenize_eval, 
         batched=False, 
         remove_columns=raw_eval_dataset.column_names
      )
   else:
      eval_dataset = None

# Create DataLoaders
   train_loader = DataLoader(
      train_dataset, 
      batch_size=train_batch_size, 
      shuffle=True, 
      collate_fn=default_data_collator
   )

   val_loader = None
   if eval_dataset:
      val_loader = DataLoader(
         eval_dataset, 
         batch_size=eval_batch_size, 
         collate_fn=collate_eval
      )

   return train_loader, val_loader