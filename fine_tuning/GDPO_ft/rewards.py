import re
import numpy as np
import time
import concurrent.futures
from fine_tuning.GDPO_ft.utils import judge_prompt_template
from models.gemini_model import gemini_model

gemini = gemini_model()

def format_complexity_reward(completions, length_cap=20, **kwargs):
    """
    Tiered Reward:
    1.  Base: Finds <question>...</question> tags (Essential).
    2.  Bonus: Finds </think> tag (Encourages thinking).
    3.  Quality: Complexity of the question text.
    """
    rewards = []

    for completion in completions:
        c = completion.strip()
        total_score = 0.0
        
        # First check for Question Tags
        question_match = re.search(r"<question>(.*?)</question>", c, re.DOTALL)
        
        if question_match:
            total_score += 0.3
            
            content = question_match.group(1).strip()
            
            # The Complexity Score (Up to 0.4)
            if content:
                # Split based on numbered list
                questions_list = re.split(r'(?:^|\n)\d+\.\s*', content)
                questions_list = [q.strip() for q in questions_list if q.strip()]
                
                if questions_list:
                    item_scores = []
                    for q in questions_list:
                        q_lower = q.lower()
                        # Length Score
                        l_score = min(len(q_lower.split()) / length_cap, 1.0)
                        # Keyword Score
                        if any(w in q_lower for w in ["why", "how", "describe", "explain", "compare", "context"]):
                            k_score = 1.0
                        else:
                            k_score = 0.5
                        item_scores.append((l_score + k_score) / 2)
                    
                    if item_scores:
                        # Add up to 0.4 based on quality
                        total_score += (np.mean(item_scores) * 0.4)

            # Thinking Bonus (Up to 0.3)
            # We look for </think> because Qwen often skips the opening tag
            if "</think>" in c:
                total_score += 0.3
        
        else:
            # If it failed to output question tags, push in right direction
            if "<think>" in c or "</think>" in c or "<question>" in c:
                total_score = 0.1  # Tiny reward to keep gradients flowing
            else:
                total_score = 0.0

        # Hard cap at 1.0 just in case
        rewards.append(min(total_score, 1.0))
        
    return rewards


# Parallel execution helper
def call_api(index, prompt_text):
    """
    Helper function to call Gemini API with retries for rate limits
    """
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            # This calls .invoke() method
            response_text = gemini.invoke(prompt_text)
            return index, response_text
        except Exception as e:
            # Check for rate limit error codes (usually 429)
            if "429" in str(e) or "ResourceExhausted" in str(e):
                wait_time = 2 ** attempt
                print(f"Rate limited on index {index}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Actual error
                print(f"Gemini API Error on index {index}: {e}")
                return index, None
    print(f"Failed to get response for index {index} after {retry_attempts} attempts.")
    return index, None


def gemini_judge_reward(completions, questions, structured_details, **kwargs):
    """
    Gemini API Judge: Parallel evaluation of questions
    """
    # Prepare all prompts first
    prompts_map = {} # Maps index -> prompt_string
    
    for i, (completion, gold_qs, context) in enumerate(zip(completions, questions, structured_details)):
        match = re.search(r"<question>(.*?)</question>", completion, re.DOTALL)
        
        if not match:
            # Format error = 0.0 immediately, no wasting API call
            continue
            
        generated_qs = match.group(1).strip()

        prompt = judge_prompt_template(
            context=context,
            gold_questions=gold_qs, 
            questions=generated_qs
        )
        prompts_map[i] = prompt

    # If no valid prompts, return zeros
    if not prompts_map:
        return [0.0] * len(completions)

    # Parallel Execution
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_idx = {}
        for idx, prompt in prompts_map.items():
            time.sleep(0.1) # Small delay to avoid hitting rate limits
            future_to_idx[executor.submit(call_api, idx, prompt)] = idx
        
        # Collect results as they finish
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, response_text = future.result()
            if response_text:
                results[idx] = response_text
            else:
                print(f"Failed response for index {idx}")
                results[idx] = None

    # Parse Results
    final_rewards = [0.0] * len(completions)
    
    for idx, response in results.items():
        if response is None:
            final_rewards[idx] = 0.0
            continue

        clean_response = response.strip()
        
        # Parse "1 0 1 1" pattern
        matches = re.search(r"([01])\D*([01])\D*([01])\D*([01])", clean_response)
        if matches:
            scores = [int(matches.group(k)) for k in range(1, 5)]
            final_rewards[idx] = sum(scores) / 4.0
        else:
            final_rewards[idx] = 0.0

    return final_rewards