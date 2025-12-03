import re
import numpy as np
import torch
from tqdm import tqdm
import random

def calculate_exact_match(preds, refs):
    """
    Returns exact match accuracy. 
    Strict comparison, usually near 0 for long generated text.
    """
    return np.mean([1 if p.strip().lower() == r.strip().lower() else 0 for p, r in zip(preds, refs)])


def calculate_f1(preds, refs):
    """
    Returns f1 score based on token overlap.
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
    Evaluates the model on the validation set.
    """
    model.eval()
    
    # Ensure padding is on the left, otherwise generation will be garbled for batched inputs
    tokenizer.padding_side = "left" 
    
    # Llama 3 uses a specific token for "End of Turn" (<|eot_id|>)
    # We must stop on this, or the regular EOS token.
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    preds, refs = [], []

    print(f"Starting evaluation on {device}...")
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_texts = batch["target_text"]  # Ground truth list of questions

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators, # Stop on <|eot_id|> or <|end_of_text|>
                do_sample=False, # Use greedy decoding for reproducible evaluation
                temperature=None,
                top_p=None
            )

        # Process batch
        for i in range(input_ids.shape[0]):
            # Slice only the generated tokens (exclude input prompt)
            # input_ids.shape[1] is the length of the prompt (including padding)
            generated_ids = outputs[i, input_ids.shape[1]:]
            
            pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Reference
            ref = target_texts[i].strip()
            
            # Clean up artifact logic if necessary (though tokenizer.decode handles most)
            # Remove "assistant" header if it leaks (rare in Llama 3 specific templates but possible)
            pred = re.sub(r"^(assistant[:\s]*)", "", pred, flags=re.IGNORECASE).strip()

            preds.append(pred)
            refs.append(ref)

    # Metrics
    exact_match = calculate_exact_match(preds, refs)
    f1 = calculate_f1(preds, refs)

    print(f"\nResults:")
    print(f"Exact Match Accuracy: {exact_match:.4f}")
    print(f"Token-level F1 Score: {f1:.4f}")

    # Sample outputs
    if show_samples > 0 and len(preds) > 0:
        print("\n" + "="*80)
        print("SAMPLE PREDICTIONS")
        print("="*80)
        random.seed(seed)
        
        # Pick random indices
        indices = random.sample(range(len(preds)), min(show_samples, len(preds)))
        
        for idx in indices:
            print(f"\n[Example {idx}]")
            print(f"--- Gold Reference ---")
            print(refs[idx])
            print(f"--- Model Prediction ---")
            print(preds[idx])
            print("-" * 40)

    return {
        "exact_match_accuracy": exact_match, 
        "f1": f1
    }