import re
import numpy as np
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
    format_pattern = r"^<think>(?s:.*?)</think>\s*<question>(?s:.*?)</question>\s*$"

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