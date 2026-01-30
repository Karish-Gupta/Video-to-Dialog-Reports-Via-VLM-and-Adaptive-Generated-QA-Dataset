import re 

def complexity_reward(completions, **kwargs):
    """
    Penalizes short questions or simple Yes/No structures
    """
    rewards = []
    for completion in completions:
        q_match = re.search(r"<question>(.*?)</question>", completion, re.DOTALL)
        if not q_match:
            rewards.append(0.0)
            continue
        
        question = q_match.group(1).lower()
        
        # Reward 1: Length heuristic (reward longer questions more)
        length_score = min(len(question.split()) / 20.0, 1.0) # Cap at 20 words
        
        # Reward 2: Keyword bonus for "Why", "How", "Describe" vs "Is", "Did"
        if any(w in question for w in ["why", "how", "explain", "describe"]):
            keyword_score = 1.0
        else:
            keyword_score = 0.5
            
        # Combine
        rewards.append((length_score + keyword_score) / 2)
        
    return rewards

def format_reward(completions, **kwargs):
    """
    Reward 1.0 if the completion strictly follows the XML format.
    """
    pattern = r"^<think>(?s:.*?)</think>\s*<question>(?s:.*?)</question>$"
    rewards = []
    for completion in completions:
        # strict matching of the tags
        if re.match(pattern, completion.strip()):
            rewards.append(1.0)
        else:
            rewards.append(0.0) 
    return rewards