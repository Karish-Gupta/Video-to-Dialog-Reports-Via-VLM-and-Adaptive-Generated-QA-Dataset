from sentence_transformers import SentenceTransformer, util
import numpy as np
import re 

def complexity_reward(completions, length_cap=20):
    """
    Evaluates the complexity of EACH item in the numbered list inside the tag.
    """
    rewards = []
    for completion in completions:
        # Extract the content inside <question> tags
        match = re.search(r"<question>(.*?)</question>", completion, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue
        
        content = match.group(1).strip()
        
        # Split by numbered list pattern to get individual questions
        questions_list = re.split(r'\n\d+\.\s*', content)
        # Remove empty strings from split
        questions_list = [q.strip() for q in questions_list if q.strip()]
        
        if not questions_list:
            rewards.append(0.0)
            continue

        item_scores = []
        for q in questions_list:
            q_lower = q.lower()
            
            # 1. Length Score (per question, not total)
            l_score = min(len(q_lower.split()) / length_cap, 1.0)
            
            # 2. Keyword Score
            if any(w in q_lower for w in ["why", "how", "describe", "explain", "compare", "context"]):
                k_score = 1.0
            else:
                k_score = 0.5
            
            item_scores.append((l_score + k_score) / 2)
        
        # Final reward is the average quality of the 4 questions
        rewards.append(np.mean(item_scores))
        
    return rewards


similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def cot_similarity_reward(completions, gold_CoT):
    """
    Reward for reasoning similarity: Compares generated <think> content vs. Gold 'CoT' column
    """
    clean_generations = []
    
    for completion in completions:
        # Extract thinking block
        match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if match:
            clean_generations.append(match.group(1).strip())
        else:
            clean_generations.append("") 
            
    # Encode and compare
    gen_embeddings = similarity_model.encode(clean_generations, convert_to_tensor=True)
    gold_embeddings = similarity_model.encode(gold_CoT, convert_to_tensor=True)
    
    scores = util.paired_cosine_similarities(gen_embeddings, gold_embeddings)
    
    return [max(0.0, score.item()) for score in scores]


def question_similarity_reward(completions, gold_questions):
    """
    Reward for answer accuracy: Compares generated <question> content vs. Gold 'questions' column
    """
    clean_generations = []
    
    for completion in completions:
        # Extract the question block
        match = re.search(r"<question>(.*?)</question>", completion, re.DOTALL)
        if match:
            clean_generations.append(match.group(1).strip())
        else:
            clean_generations.append("")
            
    gen_embeddings = similarity_model.encode(clean_generations, convert_to_tensor=True)
    gold_embeddings = similarity_model.encode(gold_questions, convert_to_tensor=True)
    
    scores = util.paired_cosine_similarities(gen_embeddings, gold_embeddings)
    return [max(0.0, score.item()) for score in scores]


def format_reward(completions):
    """
    Reward 1.0 if the completion follows the XML format
    """
    pattern = r"^<think>(?s:.*?)</think>\s*<question>(?s:.*?)</question>$"
    rewards = []
    for completion in completions:
        if re.match(pattern, completion.strip()):
            rewards.append(1.0)
        else:
            rewards.append(0.0) 
    return rewards