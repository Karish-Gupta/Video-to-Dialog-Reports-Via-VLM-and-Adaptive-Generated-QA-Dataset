import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
   login(token=os.environ["HF_TOKEN"])
   
class llm:
   def __init__(self, model):
      # Configs

      # Quantization config: load in 4-bit to save VRAM
      quant_config = BitsAndBytesConfig(
         load_in_4bit=True,
         bnb_4bit_compute_dtype=torch.float16,
         bnb_4bit_use_double_quant=True,
         bnb_4bit_quant_type="nf4"
      )
      
      self.model = AutoModelForCausalLM.from_pretrained(
         model, 
         quantization_config=quant_config,
         device_map="cuda:0"
      )
      self.tokenizer = AutoTokenizer.from_pretrained(model)
      self.tokenizer.pad_token = self.tokenizer.eos_token # Set padding token for Llama
      
   
   def build_transcript_context(self, transcript):
      prompt_template = f"""
      You are summarizing a police bodycam video transcript. 
      Ensure that the summary is detailed and includes all pieces of information that could be helpful to law enforcement when making a report.

      Transcript: 
      {transcript}
      """

      return prompt_template

   
   def invoke(self, prompt):
      inputs = self.tokenizer(
         prompt, 
         return_tensors="pt", 
         truncation=True, 
         max_length=7000 # Token length for llama is 8192 for input and output
      ).to(self.model.device)
      
      outputs = self.model.generate(
         **inputs, 
         max_new_tokens=256, 
         do_sample=True,
         temperature=0.5,
         pad_token_id=self.tokenizer.pad_token_id,  # Explicit pad token
         eos_token_id=self.tokenizer.eos_token_id
      )
      
      # Decode only generated tokens
      gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
      decoded_output = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

      print(f"Generated {len(gen_tokens)} tokens") # Debug

      return decoded_output.strip()
   


