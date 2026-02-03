import re
import os
from vllm import LLM, SamplingParams
from fine_tuning.GDPO_ft.utils import JUDGE_PROMPT_TEMPLATE

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Set to available GPU to 1

# Initialize vLLM Judge
judge_model_name = "Qwen/Qwen2.5-1.5B-Instruct" # Using very small model 

llm_judge = LLM(
   model=judge_model_name,
   trust_remote_code=True,
   dtype="float16",
   tensor_parallel_size=1,
   gpu_memory_utilization=0.9
   )

judge_sampling_params = SamplingParams(
   temperature=0, 
   max_tokens=10,
   top_p=1.0
   )

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Reset for main training


# Judge reward function
def judge_reward(completions, questions, **kwargs):
   """
   vLLM Version: Batched evaluation of questions.
   """
   prompts = []
   valid_indices = []

   for i, (completion, gold_qs) in enumerate(zip(completions, questions)):
      match = re.search(r"<question>(.*?)</question>", completion, re.DOTALL)
      
      if not match:
         continue
         
      generated_qs = match.group(1).strip()
      
      user_content = JUDGE_PROMPT_TEMPLATE.format(
         gold_questions=gold_qs, 
         questions=generated_qs
      )
      
      # Use the vLLM tokenizer wrapper
      messages = [{"role": "user", "content": user_content}]
      input_text = llm_judge.get_tokenizer().apply_chat_template(
         messages, 
         tokenize=False, 
         add_generation_prompt=True
      )
      
      prompts.append(input_text)
      valid_indices.append(i)

   # Exit if no valid prompts
   if not prompts:
      print("LLM Judge Failed: No valid completions for judge reward.")
      return [0.0] * len(completions)

   # vLLM Batch Inference
   outputs = llm_judge.generate(prompts, judge_sampling_params, use_tqdm=False)

   # Parse Results
   final_rewards = [0.0] * len(completions)
   
   for idx_in_valid, output_obj in enumerate(outputs):
      original_idx = valid_indices[idx_in_valid]
      response = output_obj.outputs[0].text.strip()
      
      # Parse "1 0 1 1" pattern
      matches = re.search(r"([01])\D*([01])\D*([01])\D*([01])", response)
      if matches:
         scores = [int(matches.group(k)) for k in range(1, 5)]
         final_rewards[original_idx] = sum(scores) / 4.0
      else:
         final_rewards[original_idx] = 0.0

   return final_rewards