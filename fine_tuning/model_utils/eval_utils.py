import re
import numpy as np
import torch
from tqdm import tqdm
import random
from bert_score import score

def calculate_f1(preds, refs):
    """
    Returns f1 score based on token overlap
    """
    f1_scores = []
    for p, r in zip(preds, refs):
        ptoks = p.lower().split()
        rtoks = r.lower().split()
        
        # Avoid division by zero if empty strings
        if not ptoks or not rtoks:
            f1_scores.append(0.0)
            continue
            
        common = set(ptoks) & set(rtoks)
        num_common = sum(min(ptoks.count(t), rtoks.count(t)) for t in common)
        
        if num_common == 0:
            f1_scores.append(0.0)
        else:
            prec = num_common / len(ptoks)
            rec = num_common / len(rtoks)
            f1_scores.append(2 * prec * rec / (prec + rec))
            
    return np.mean(f1_scores) if f1_scores else 0.0

def evaluate_model(model, val_loader, device, tokenizer, max_gen_length=256, show_samples=5, seed=101):
    """
    Evaluates the model on the validation set using Token F1 and BERTScore.
    """
    model.eval()
       
    # Llama 3 specific terminators
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    preds, refs = [], []

    print(f"Starting evaluation on {device}...")
    
    for batch in tqdm(val_loader, desc="Generating Responses"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_texts = batch["target_text"] 

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
                do_sample=False, # Greedy decoding
                temperature=None,
                top_p=None
            )

        # Process batch
        for i in range(input_ids.shape[0]):
            # Slice generated tokens
            generated_ids = outputs[i, input_ids.shape[1]:]
            pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Clean "assistant" header if present
            pred = re.sub(r"^(assistant[:\s]*)", "", pred, flags=re.IGNORECASE).strip()

            ref = target_texts[i].strip()

            preds.append(pred)
            refs.append(ref)

    # --- Metrics Calculation ---

    # F1 Score Calculation
    token_f1 = calculate_f1(preds, refs)

    # BERTScore Calculation
    print("\nCalculating BERTScore...")
    try:
        # P = Precision, R = Recall, F1 = F1 Score
        P, R, F1_bert = score(
            preds, 
            refs, 
            lang="en", 
            verbose=True, 
            device=device,
            batch_size=16 
        )
        bert_score_mean = F1_bert.mean().item()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        bert_score_mean = 0.0

    print(f"\n{'-'*30}")
    print(f"RESULTS")
    print(f"Token-level F1 Score:  {token_f1:.4f}")
    print(f"BERTScore F1 (Mean):   {bert_score_mean:.4f}")
    print(f"{'-'*30}")

    # Sample outputs
    if show_samples > 0 and len(preds) > 0:
        print("\n" + "="*80)
        print("SAMPLE PREDICTIONS")
        print("="*80)
        random.seed(seed)
        
        indices = random.sample(range(len(preds)), min(show_samples, len(preds)))
        
        for idx in indices:
            print(f"\n[Example {idx}]")
            print(f"--- Gold Reference ---")
            print(refs[idx])
            print(f"--- Model Prediction ---")
            print(preds[idx])
            print("-" * 40)

    return token_f1, bert_score_mean