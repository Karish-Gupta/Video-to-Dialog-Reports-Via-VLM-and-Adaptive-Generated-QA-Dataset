import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

judge_model_name = "Qwen/Qwen2.5-1.5B-Instruct" # Using very small model 

judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
judge_model = AutoModelForCausalLM.from_pretrained(
   judge_model_name,
   torch_dtype=torch.float16,
   device_map="cuda:0",
   trust_remote_code=True
   )


JUDGE_PROMPT_TEMPLATE = """
CONTEXT (Video Details):
{context}

STUDENT QUESTIONS:
{questions}

GRADING CRITERIA:
1. Relevance: Do the questions target specific details mentioned in the context?
2. Specificity: Are they specific (e.g., "What color is the hoodie?") rather than vague ("What does he look like?")?
3. Utility: Would the answers to these questions actually help an investigation?
4. Logic: Do the questions make sense given the sequence of events?

TASK:
Rate the set of questions on a scale from 1 to 5.
Output ONLY the integer score. Do not explain.
"""

def judge_reward(completions, structured_details):
   """
   Uses a separate LLM to grade the generated questions 1-5.
   Normalizes score to 0.0 - 1.0.
   """
   rewards = []
   judge_inputs = []
   valid_indices = []

   # Prepare Prompts for the Judge
   for i, (completion, context) in enumerate(zip(completions, structured_details)):
      # Extract questions
      match = re.search(r"<question>(.*?)</question>", completion, re.DOTALL)
      
      if not match:
         rewards.append(0.0) 
         continue
         
      questions_text = match.group(1).strip()
      
      # Format the prompt for the Judge
      user_content = JUDGE_PROMPT_TEMPLATE.format(
         context=context, 
         questions=questions_text
      )
      
      messages = [{"role": "user", "content": user_content}]
      input_text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      judge_inputs.append(input_text)
      valid_indices.append(i)

   if not judge_inputs:
      return rewards

   # Batch Inference
   inputs = judge_tokenizer(judge_inputs, return_tensors="pt", padding=True, truncation=True).to(judge_model.device)
   
   with torch.no_grad():
      # Generate only 2 tokens (we just want the number)
      outputs = judge_model.generate(**inputs, max_new_tokens=2, temperature=0.1)
      generated_responses = judge_tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

   # Parse Scores
   current_valid_idx = 0
   
   # Reconstruct the full rewards list
   final_rewards = []
   
   for i in range(len(completions)):
      if i not in valid_indices:
         final_rewards.append(0.0)
      else:
         response = generated_responses[current_valid_idx].strip()
         current_valid_idx += 1
         
         # Attempt to find a digit 1-5 in the response
         score_match = re.search(r"[1-5]", response)
         if score_match:
               score = int(score_match.group(0))
               # Normalize 1-5 range (1=0.0, 2=0.25, 3=0.5, 4=0.75, 5=1.0)
               normalized_score = (score - 1) / 4.0 
               final_rewards.append(normalized_score)
         else:
               final_rewards.append(0.0)

   return final_rewards