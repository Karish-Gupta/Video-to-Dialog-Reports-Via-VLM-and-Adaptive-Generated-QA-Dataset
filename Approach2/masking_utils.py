"""
Masks subsections within structured output (such as Location_Clues -> [MASK])
"""

import json
import random
from typing import Dict, List, Any, Tuple


def mask_subsection_value(value: Any, mask_token: str = "[MASK]") -> Any:
    """
    Mask a single value (string, list, or dict) with [MASK] token.
    
    Args:
        value: The value to mask
        mask_token: Token to use for masking
        
    Returns:
        Masked value
    """
    if isinstance(value, str):
        # Only mask non-empty strings
        return mask_token if value.strip() else value
    elif isinstance(value, list):
        # Mask list by replacing with single mask token or empty list
        return [] if len(value) == 0 else [mask_token]
    elif isinstance(value, dict):
        # For nested dicts, mask all values
        return {k: mask_token for k in value.keys()}
    else:
        return mask_token


def apply_masking_strategy(
    structured_data: Dict[str, Any],
    mask_probability: float = 0.3,
    min_masks: int = 1,
    max_masks: int = 5,
    seed: int = None
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply masking strategy to structured data.
    Randomly mask subsection values within Scene-Level, Entity-Level, Action-Level, Semantic-Level.
    
    Args:
        structured_data: Parsed structured details dictionary
        mask_probability: Probability of masking each field
        min_masks: Minimum number of fields to mask
        max_masks: Maximum number of fields to mask
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (masked_data, list of masked field paths)
    """
    if seed is not None:
        random.seed(seed)
    
    masked_data = json.loads(json.dumps(structured_data))  # Deep copy
    masked_fields = []
    
    # Define the main sections
    main_sections = ["Scene-Level", "Entity-Level", "Action-Level", "Semantic-Level"]
    
    # Collect all maskable paths
    maskable_paths = []
    
    for section in main_sections:
        if section not in structured_data:
            continue
            
        section_data = structured_data[section]
        if not isinstance(section_data, dict):
            continue
        
        for key, value in section_data.items():
            # We can mask individual fields
            if isinstance(value, (str, list)):
                # Skip if already empty
                if (isinstance(value, str) and not value.strip()) or \
                   (isinstance(value, list) and len(value) == 0):
                    continue
                maskable_paths.append((section, key, False))  # False = not nested
            elif isinstance(value, dict):
                # For nested structures (like People list with dicts)
                maskable_paths.append((section, key, True))  # True = nested
    
    # Determine how many fields to mask
    num_to_mask = random.randint(min_masks, min(max_masks, len(maskable_paths)))
    
    # Select fields to mask
    if len(maskable_paths) > 0:
        fields_to_mask = random.sample(maskable_paths, num_to_mask)
        
        for section, key, is_nested in fields_to_mask:
            if is_nested:
                # For nested structures, mask specific sub-fields
                value = masked_data[section][key]
                if isinstance(value, list):
                    # Mask items in list
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            # Randomly select sub-keys to mask
                            sub_keys = list(item.keys())
                            if sub_keys:
                                keys_to_mask = random.sample(
                                    sub_keys, 
                                    random.randint(1, len(sub_keys))
                                )
                                for sub_key in keys_to_mask:
                                    masked_data[section][key][i][sub_key] = "[MASK]"
                                    masked_fields.append(f"{section}.{key}[{i}].{sub_key}")
                elif isinstance(value, dict):
                    # Mask values in dict
                    sub_keys = list(value.keys())
                    if sub_keys:
                        keys_to_mask = random.sample(
                            sub_keys,
                            random.randint(1, len(sub_keys))
                        )
                        for sub_key in keys_to_mask:
                            masked_data[section][key][sub_key] = "[MASK]"
                            masked_fields.append(f"{section}.{key}.{sub_key}")
            else:
                # Simple field masking
                masked_data[section][key] = mask_subsection_value(
                    masked_data[section][key]
                )
                masked_fields.append(f"{section}.{key}")
    
    return masked_data, masked_fields


def create_masked_dataset(
    input_jsonl_path: str,
    output_jsonl_path: str,
    mask_probability: float = 0.3,
    min_masks: int = 1,
    max_masks: int = 5,
    seed: int = 42
):
    """
    Process the entire data.jsonl file and create a masked version.
    
    Args:
        input_jsonl_path: Path to input data.jsonl
        output_jsonl_path: Path to output masked data.jsonl
        mask_probability: Probability of masking each field
        min_masks: Minimum number of fields to mask per example
        max_masks: Maximum number of fields to mask per example
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    masked_examples = []
    
    with open(input_jsonl_path, 'r') as f:
        for line_idx, line in enumerate(f):
            example = json.loads(line)
            
            # Parse structured details - strip markdown code blocks
            structured_str = example['structured_details'].strip()
            if structured_str.startswith("```"):
                lines = structured_str.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                structured_str = '\n'.join(lines)
            
            # Parse JSON
            try:
                structured_data = json.loads(structured_str)
            except Exception as e:
                print(f"Warning: Could not parse structured_details for line {line_idx}: {e}")
                continue
            
            # Apply masking
            masked_data, masked_fields = apply_masking_strategy(
                structured_data,
                mask_probability=mask_probability,
                min_masks=min_masks,
                max_masks=max_masks,
                seed=seed + line_idx  # Vary seed per example
            )
            
            # Create new example with masked data - keep as clean JSON string
            masked_example = {
                'video_index': example['video_index'],
                'vlm_summary': example['vlm_summary'],
                'structured_details': json.dumps(masked_data, indent=2),
                'masked_fields': masked_fields,  # Track what was masked
                'questions': example['questions']
            }
            
            masked_examples.append(masked_example)
    
    # Write to output file
    with open(output_jsonl_path, 'w') as f:
        for example in masked_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created {len(masked_examples)} masked examples")
    print(f"Output written to: {output_jsonl_path}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Create masked dataset for Approach2")
    parser.add_argument("--input", type=str, required=True, help="Input data.jsonl path")
    parser.add_argument("--output", type=str, required=True, help="Output masked data.jsonl path")
    parser.add_argument("--mask_prob", type=float, default=0.3, help="Masking probability")
    parser.add_argument("--min_masks", type=int, default=1, help="Minimum masks per example")
    parser.add_argument("--max_masks", type=int, default=5, help="Maximum masks per example")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    create_masked_dataset(
        input_jsonl_path=args.input,
        output_jsonl_path=args.output,
        mask_probability=args.mask_prob,
        min_masks=args.min_masks,
        max_masks=args.max_masks,
        seed=args.seed
    )
