import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
   login(token=os.environ["HF_TOKEN"])
   
class llm:
   def __init__(self, model):
      # Configs
      self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16, device_map="cuda:0")
      self.tokenizer = AutoTokenizer.from_pretrained(model)

   def report_prompt(self, text):
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
   
   def build_transcript_context(self, transcript):
      prompt_template = f"""
      You are summarizing a police bodycam video transcript. 
      Ensure that the summary is detailed and includes all pieces of information that could be helpful to law enforcement when making a report.

      Transcript: 
      {transcript}
      """

      return prompt_template

   
   def invoke(self, prompt):
      inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
      outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=1)
      decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
      return decoded_output

