import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
   login(token=os.environ["HF_TOKEN"])
   
class llm:
   def __init__(self, model_name):
      # Configs
      self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   def first_run(self, text):
      prompt_template = f"""
         You are generating training data for a vision-language model. 
         Given the structured scene summary below format the text into this structured format:

         Structured summary:
         {{
            "Observation": "...",
            "Action": "...",
            "Intent/Reason": "...",
            "Response": "...",
            "Inference": "...",
            "Individuals Involved": "...",
            "Context Summary": "..."
         }}

         Text: {text}
      """

      return prompt_template

   
   def invoke(self, prompt):
      inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
      outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=1)
      print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))