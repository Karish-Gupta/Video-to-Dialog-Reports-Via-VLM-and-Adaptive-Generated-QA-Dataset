import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

def collate_train(batch, pad_token_id):
   """
   Collate function for training that properly pads sequences to max length in batch.
   """
   input_ids_list = []
   attention_mask_list = []
   labels_list = []
   
   # First pass: convert all to lists and collect lengths
   for b in batch:
      # Robust conversion to list
      input_ids = b["input_ids"]
      if isinstance(input_ids, torch.Tensor):
         input_ids = input_ids.tolist()
      elif isinstance(input_ids, np.ndarray):
         input_ids = input_ids.tolist()
      elif not isinstance(input_ids, list):
         input_ids = list(input_ids)
      
      attention_mask = b["attention_mask"]
      if isinstance(attention_mask, torch.Tensor):
         attention_mask = attention_mask.tolist()
      elif isinstance(attention_mask, np.ndarray):
         attention_mask = attention_mask.tolist()
      elif not isinstance(attention_mask, list):
         attention_mask = list(attention_mask)
      
      labels = b["labels"]
      if isinstance(labels, torch.Tensor):
         labels = labels.tolist()
      elif isinstance(labels, np.ndarray):
         labels = labels.tolist()
      elif not isinstance(labels, list):
         labels = list(labels)
      
      input_ids_list.append(input_ids)
      attention_mask_list.append(attention_mask)
      labels_list.append(labels)
   
   # Find max sequence length
   max_seq_len = max(len(seq) for seq in input_ids_list)
   
   # Pad all sequences to max length
   padded_input_ids = []
   padded_attention_mask = []
   padded_labels = []
   
   for input_ids, attention_mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
      pad_len = max_seq_len - len(input_ids)
      
      padded_input_ids.append(input_ids + [pad_token_id] * pad_len)
      padded_attention_mask.append(attention_mask + [0] * pad_len)
      padded_labels.append(labels + [-100] * pad_len)
   
   # Verify all sequences have same length before converting to tensor
   assert all(len(seq) == max_seq_len for seq in padded_input_ids), \
      f"Input IDs have mismatched lengths: {[len(seq) for seq in padded_input_ids]}"
   assert all(len(seq) == max_seq_len for seq in padded_attention_mask), \
      f"Attention masks have mismatched lengths: {[len(seq) for seq in padded_attention_mask]}"
   assert all(len(seq) == max_seq_len for seq in padded_labels), \
      f"Labels have mismatched lengths: {[len(seq) for seq in padded_labels]}"
   
   return {
      "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
      "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
      "labels": torch.tensor(padded_labels, dtype=torch.long),
   }

def collate_eval(batch):
   """
   Collate function that keeps the target_text as a list for metric computation,
   while batching input_ids/attention_mask for the model.
   """
   pad_token_id = 0  # Default padding token ID
   
   # Convert all sequences to lists (handle numpy arrays, tensors, etc.)
   input_ids_list = []
   attention_mask_list = []
   target_texts = []
   
   for b in batch:
      # Convert to list if necessary
      input_ids = b["input_ids"]
      if not isinstance(input_ids, list):
         input_ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
      
      attention_mask = b["attention_mask"]
      if not isinstance(attention_mask, list):
         attention_mask = attention_mask.tolist() if hasattr(attention_mask, 'tolist') else list(attention_mask)
      
      input_ids_list.append(input_ids)
      attention_mask_list.append(attention_mask)
      target_texts.append(b["target_text"])
   
   # Find max sequence length after conversion
   max_seq_len = max(len(seq) for seq in input_ids_list)
   
   # Pad all sequences to max length
   padded_input_ids = []
   padded_attention_mask = []
   
   for i in range(len(input_ids_list)):
      padding_len = max_seq_len - len(input_ids_list[i])
      
      padded_input_ids.append(input_ids_list[i] + [pad_token_id] * padding_len)
      padded_attention_mask.append(attention_mask_list[i] + [0] * padding_len)
   
   return {
      "input_ids": torch.tensor(padded_input_ids),
      "attention_mask": torch.tensor(padded_attention_mask),
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
   Preprocess dataset for instruction fine-tuning (Qwen3 / Llama 3 compatible).
   Uses the model's native chat template via apply_chat_template().
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
   
   # Note: apply_chat_template() is compatible with both Qwen3 and Llama 3
   # as long as the tokenizer is loaded from the correct model

   # Tokenize Function for Training
   def tokenize_train(example):
      # Construct the conversation (Qwen3 and Llama 3 compatible)
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
      collate_fn=lambda batch: collate_train(batch, tokenizer.pad_token_id)
   )

   val_loader = None
   if eval_dataset:
      val_loader = DataLoader(
         eval_dataset, 
         batch_size=eval_batch_size, 
         collate_fn=collate_eval
      )

   return train_loader, val_loader