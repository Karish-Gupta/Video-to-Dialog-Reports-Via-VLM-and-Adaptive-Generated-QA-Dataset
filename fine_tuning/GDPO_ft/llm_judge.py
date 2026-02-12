import re
import os
from vllm import LLM, SamplingParams
from fine_tuning.GDPO_ft.utils import judge_prompt_template

class LLMJudgeReward:

   _engine = None
   _tokenizer = None

   def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", gpu_id="1"):
      self.model_name = model_name
      self.gpu_id = gpu_id

      self.judge_params = SamplingParams(
         temperature=0, 
         max_tokens=10,
         top_p=1.0
      )
   
   def _get_engine(self):
      if LLMJudgeReward._engine is None:
         print("Initializing LLM Judge Engine...")

         # Set GPU for judge, mask original GPUs
         original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
         os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id

         # Intialize vLLM engine
         LLMJudgeReward._engine = LLM(
            model=self.model_name,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
         )

         # Intialize vLLM tokenizer
         LLMJudgeReward._tokenizer = LLMJudgeReward._engine.get_tokenizer()
         
         # Restore original visible devices
         if original_visible_devices:
            print("Restoring original CUDA_VISIBLE_DEVICES...")
            os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices

      return LLMJudgeReward._engine, LLMJudgeReward._tokenizer


   # Judge reward function
   def __call__(self, completions, questions, structured_details, **kwargs):
      """
      vLLM Version: Batched evaluation of questions.
      """
      llm_judge, llm_tokenizer = self._get_engine()

      prompts = []
      valid_indices = []

      for i, (completion, gold_qs, context) in enumerate(zip(completions, questions, structured_details)):
         match = re.search(r"<question>(.*?)</question>", completion, re.DOTALL)
         
         if not match:
            continue
            
         generated_qs = match.group(1).strip()
         
         user_content = judge_prompt_template(context, gold_qs, generated_qs)
         
         # Use the vLLM tokenizer wrapper
         messages = [{"role": "user", "content": user_content}]
         input_text = llm_tokenizer.apply_chat_template(
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
      outputs = llm_judge.generate(prompts, self.judge_params, use_tqdm=False)

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