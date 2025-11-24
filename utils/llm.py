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
      system_prompt = (
        "You analyze police body-worn camera recordings.\n"
        "You will be given:\n"
        "  - A visual summary of the video (<summary>)\n"
        "  - The transcript of spoken dialogue (<transcript>)\n\n"
        "Your task is to extract factual key details strictly grounded in what can be:\n"
        "  - Seen\n"
        "  - Heard\n"
        "  - Directly inferred from observable physical evidence\n\n"
        "Do NOT speculate or invent missing details.\n"
        "Return ONLY the JSON output structure—no commentary, no explanation."
    )

      user_prompt = f"""
      <summary>
      {json.dumps(summary)}
      </summary>

      <transcript>
      {json.dumps(transcript)}
      </transcript>

      Extract and output key details using the following structure:

      {{
      "Scene-Level": {{
         "Environment": "",        // Indoors/outdoors, setting type, weather, lighting
         "Location_Clues": "",     // Visible signage, street names, inferred setting only if visually grounded
         "Scene_Changes": []       // Changes in environment or camera movement (entry/exit rooms, approach vehicle, etc.)
      }},
      
      "Entity-Level": {{
         "People": [
            {{
            "Description": "",     // Clothing, identifiers, notable appearance features
            "Role_if_Clear": "",   // officer, civilian, suspect (ONLY if visually or transcript-confirmed)
            "Position": ""         // relative spatial location (left/right/behind/near doorway/etc.)
            }}
         ],
         "Animals": [],
         "Objects": [
            {{
            "Type": "",
            "Location": "",
            "Attributes": ""       // visible characteristics: damaged, brand, color, shape
            }}
         ]
      }},

      "Action-Level": {{
         "Primary_Actions": [],     // major events or movements observed in order
         "Secondary_Actions": [],   // gestures, handling items, positioning, approach/retreat
         "Interactions": []         // human-object, human-human, object-object
      }},

      "Semantic-Level": {{
         "Intent_if_Visible": "",   // ONLY if visually clear (e.g., fleeing, surrendering, reaching for object)
         "Emotional_State": "",     // body language, tone indicators (NOT speculation)
         "Notable_Audio": []        // shouting, sirens, arguments, commands, unknown sounds
      }}
      }}

      Rules:
      - If a category has no evidence, return an empty string or empty list.
      - Do NOT repeat transcript verbatim — summarize into structured facts.
      """
      
      messages = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}
      ]

      prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      return prompt
   
   
   def step_2_chat_template(self, structured_output):
      system_prompt = """
         You analyze police body-worn camera recordings.
         Generate 4 specific questions based on key details that would help clarify or expand understanding of the scene.
         Ensure questions are directly relevant to observable details from the structured output.
         Return ONLY the questions as a numbered list—no commentary, no explanation.
      """
      
      user_prompt = f"Key Details:\n {structured_output}"

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
   
   def eval_chat_template(self, caption, ground_truth, eval_prompt_template):
      system_prompt = f"{eval_prompt_template}"
      
      user_prompt = f"""
         Ground Truth:
         {ground_truth}

         Model Caption:
         {caption}
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
   


