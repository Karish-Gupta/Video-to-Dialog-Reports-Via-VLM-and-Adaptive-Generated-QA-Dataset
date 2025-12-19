import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
   login(token=os.environ["HF_TOKEN"])
   
class distillation_ft_llm:
    def __init__(self):
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        adapter_path = "llama3-8b-instruct-police-questions-lora-gemini-vlm"
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="cuda:0"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path) 
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval() # Good practice to set to eval model
        
    def step_2_chat_template(self, structured_output):
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
        
        user_prompt = f"Structured information provided:\n {structured_output}"

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
            max_length=8000 # Llama 3 context is 8192
        ).to(self.model.device)
        
        # Define terminators to stop generation correctly
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=1024, 
                do_sample=True,
                temperature=0.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=terminators # Use the custom terminators list
            )
        
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        decoded_output = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        return decoded_output.strip()