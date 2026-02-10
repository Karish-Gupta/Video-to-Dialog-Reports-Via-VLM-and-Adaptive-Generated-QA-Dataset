import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

def collate_train(batch, pad_token_id):
   """
   Collate function for training that properly pads sequences to max length in batch.
   """
   max_seq_len = max(len(b["input_ids"]) for b in batch)
   
   input_ids_list = []
   attention_mask_list = []
   labels_list = []
   
   for b in batch:
      seq_len = len(b["input_ids"])
      padding_len = max_seq_len - seq_len
      
      # Pad input_ids
      padded_input_ids = b["input_ids"] + [pad_token_id] * padding_len
      input_ids_list.append(padded_input_ids)
      
      # Pad attention_mask
      padded_attention_mask = b["attention_mask"] + [0] * padding_len
      attention_mask_list.append(padded_attention_mask)
      
      # Pad labels
      padded_labels = b["labels"] + [-100] * padding_len
      labels_list.append(padded_labels)
   
   return {
      "input_ids": torch.tensor(input_ids_list),
      "attention_mask": torch.tensor(attention_mask_list),
      "labels": torch.tensor(labels_list),
   }

def collate_eval(batch):
   """
   Collate function that keeps the target_text as a list for metric computation,
   while batching input_ids/attention_mask for the model.
   """
   max_seq_len = max(len(b["input_ids"]) for b in batch)
   pad_token_id = 0  # Default padding token ID
   
   input_ids_list = []
   attention_mask_list = []
   target_texts = []
   
   for b in batch:
      seq_len = len(b["input_ids"])
      padding_len = max_seq_len - seq_len
      
      # Pad input_ids
      padded_input_ids = b["input_ids"] + [pad_token_id] * padding_len
      input_ids_list.append(padded_input_ids)
      
      # Pad attention_mask
      padded_attention_mask = b["attention_mask"] + [0] * padding_len
      attention_mask_list.append(padded_attention_mask)
      
      target_texts.append(b["target_text"])
   
   return {
      "input_ids": torch.tensor(input_ids_list),
      "attention_mask": torch.tensor(attention_mask_list),
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