import json
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

      # Quantization config: load in 4-bit to save VRAM
      quant_config = BitsAndBytesConfig(
         load_in_4bit=True,
         bnb_4bit_compute_dtype=torch.float16,
         bnb_4bit_use_double_quant=False,
         bnb_4bit_quant_type="nf4"
      )
      
      self.model = AutoModelForCausalLM.from_pretrained(
         model, 
         quantization_config=quant_config,
         device_map="cuda:0"
      )
      self.tokenizer = AutoTokenizer.from_pretrained(model)
      self.tokenizer.pad_token = self.tokenizer.eos_token # Set padding token for Llama
      
   
   def step_1_chat_template(self, transcript, summary):
      # Use chat template for step 1 prompt
      system_prompt = "You are working with bodycam video transcript information. You are given a police bodycam transcript inside <transcript> tags and a visual summary in <summary> tags. Extract key details and return ONLY key details in valid JSON."
      
      user_prompt = f"""
         <summary>
         {json.dumps(summary)}
         <summary>
         
         <transcript>
         {json.dumps(transcript)}
         </transcript>

         Output JSON structure:
         {{
         "Scene Observations": [""],
         "Actions": [""],
         "Items in Frame": [""],
         "Descriptions of Idividuals in Frame": [""],
         }}
      """
      
      messages = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}
      ]

      prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      return prompt
   
   
   def step_2_chat_template(self, structured_output):
      system_prompt = """
         Based on the given structured information about a police bodycam video, generate specific questions based on pair combinations of key detail elements:
      
         Questions to generate:
         1. Scene Observations  
         2. Items in Frame  
         3. Descriptions of Idividuals in Frame
         4. Actions 

         Examples:
         1. Why is the vehicle pulled over along the side of the road? What is the passenger holding in their hands?
         2. What items are in the suspect's car? What is the officer holding?
         3. What clothing or accessories is the suspect wearing? What is the age, ethnicity, and gender of the suspect?
         4. Why is the officer yelling profanity at the suspect? What is the officer doing to the suspect?
      """
      
      user_prompt = f"Structured information:\n {structured_output}"

      messages = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}
      ]

      prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      return prompt
   
   def qa_caption_chat_template(self, questions, answers, transcript, vlm_summary):
      system_prompt = f"""
         You are given a bodycam video transcript, visual summary, and question-answer pairs.
         Generate a caption that gives visual details about the video. 
         Ensure that you make use of the questions and answers to enhance the caption.
         Include the following in caption: 

         - Describe the setting (Time of day, vehicles, buildings, etc.)
         - Objects in the frame (Weapons, items in hand, consumables, etc.)
         - Describe how items are being used (Is a weapon being fired, radio being held by officer, etc.)
         - Describe individuals (What are people wearing, color of vehicles, accessory items worn such as hats or glasses, etc.)
         - Actions each individual made (Officer stating instructions, civilians complying, etc.)

         Write in active voice as much as possible.
         Be direct, concise, and concrete.
         Use direct quotes only when needed.
         Use a person's name if it is known.
      """
               
      user_prompt = f"""
         Transcript: 
         {transcript}
         
         Visual Summary:
         {vlm_summary}
         
         Questions:
         {questions}
         
         Answers:
         {answers}
      """
      
      messages = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}
      ]

      prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      return prompt
   
   def caption_chat_template(self, transcript, vlm_summary):
      system_prompt = f"""
         You are given a bodycam video transcript, visual summary.
         Generate a caption that gives visual details about the video. 
         Include the following in caption: 

         - Describe the setting (Time of day, vehicles, buildings, etc.)
         - Objects in the frame (Weapons, items in hand, consumables, etc.)
         - Describe how items are being used (Is a weapon being fired, radio being held by officer, etc.)
         - Describe individuals (What are people wearing, color of vehicles, accessory items worn such as hats or glasses, etc.)
         - Actions each individual made (Officer stating instructions, civilians complying, etc.)

         Ensure captions are direct and formal.

         Write in active voice as much as possible.
         Be direct, concise, and concrete.
         Use direct quotes only when needed.
         Use a person's name if it is known.
      """
      
      user_prompt = f"""
         Transcript: 
         {transcript}
         
         Visual Summary:
         {vlm_summary}
      """
      
      messages = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}
      ]

      prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      return prompt
   
   def invoke(self, prompt):
      inputs = self.tokenizer(
         prompt, 
         return_tensors="pt", 
         truncation=True, 
         max_length=7000 # Token length for llama is 8192 for input and output
      ).to(self.model.device)
      
      outputs = self.model.generate(
         **inputs, 
         max_new_tokens=1024, 
         do_sample=True,
         temperature=0.2,
         pad_token_id=self.tokenizer.pad_token_id,  # Explicit pad token
         eos_token_id=self.tokenizer.eos_token_id
      )
      
      # Decode only generated tokens
      gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
      decoded_output = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

      print(f"Generated {len(gen_tokens)} tokens") # Debug

      return decoded_output.strip()
   


