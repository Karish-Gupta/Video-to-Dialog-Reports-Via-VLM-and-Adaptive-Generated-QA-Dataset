import re
import numpy as np
import time
import concurrent.futures
from fine_tuning.GDPO_ft.utils import JUDGE_PROMPT_TEMPLATE
from models.gemini_model import gemini_model

# from sentence_transformers import SentenceTransformer, util

# similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# def cot_similarity_reward(completions, gold_CoT):
#     """
#     Reward for reasoning similarity: Compares generated <think> content vs. Gold 'CoT' column
#     """
#     clean_generations = []
    
#     for completion in completions:
#         # Extract thinking block
#         match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
#         if match:
#             clean_generations.append(match.group(1).strip())
#         else:
#             clean_generations.append("") 
            
#     # Encode and compare
#     gen_embeddings = similarity_model.encode(clean_generations, convert_to_tensor=True)
#     gold_embeddings = similarity_model.encode(gold_CoT, convert_to_tensor=True)
    
#     scores = util.paired_cosine_similarities(gen_embeddings, gold_embeddings)
    
#     return [max(0.0, score.item()) for score in scores]


# def question_similarity_reward(completions, gold_questions):
#     """
#     Reward for answer accuracy: Compares generated <question> content vs. Gold 'questions' column
#     """
#     clean_generations = []
    
#     for completion in completions:
#         # Extract the question block
#         match = re.search(r"<question>(.*?)</question>", completion, re.DOTALL)
#         if match:
#             clean_generations.append(match.group(1).strip())
#         else:
#             clean_generations.append("")
            
#     gen_embeddings = similarity_model.encode(clean_generations, convert_to_tensor=True)
#     gold_embeddings = similarity_model.encode(gold_questions, convert_to_tensor=True)
    
#     scores = util.paired_cosine_similarities(gen_embeddings, gold_embeddings)
#     return [max(0.0, score.item()) for score in scores]


def format_complexity_reward(completions, length_cap=20, **kwargs):
    """
    Merged Reward: 
    - 50% score for perfect XML structure <think>...</think><question>...</question>
    - 50% score for the complexity of the questions inside
    """
    rewards = []
    
    # Enforce regex pattern for the structure
    format_pattern = r"</think>.*?<question>(.+?)</question>"    

    for completion in completions:
        c = completion.strip()
        
        # Calculate format Score
        if re.match(format_pattern, c):
            format_score = 1.0
        else:
            format_score = 0.0

        # Calculate complexity Score
        complexity_score = 0.0
        
        # Extract content inside <question> tags
        match = re.search(r"<question>(.*?)</question>", c, re.DOTALL)
        
        if match:
            content = match.group(1).strip()
            
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
                
                complexity_score = np.mean(item_scores)
        
        total_score = (format_score * 0.5) + (complexity_score * 0.5)
        rewards.append(total_score)
        
    return rewards


# Parallel execution helper
def call_api(index, prompt_text):
    """
    Helper function to call Gemini API with retries for rate limits
    """
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            # This calls your .invoke() method
            response_text = gemini_model.invoke(prompt_text)
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


def gemini_judge_reward(completions, questions, **kwargs):
    """
    Gemini API Judge: Parallel evaluation of questions
    """
    # Prepare all prompts first
    prompts_map = {} # Maps index -> prompt_string
    
    for i, (completion, gold_qs) in enumerate(zip(completions, questions)):
        match = re.search(r"<question>(.*?)</question>", completion, re.DOTALL)
        
        if not match:
            # Format error = 0.0 immediately, no wasting API call
            continue
            
        generated_qs = match.group(1).strip()

        prompt = JUDGE_PROMPT_TEMPLATE.format(
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