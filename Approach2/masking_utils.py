"""
Masks subsections within structured output based on what the gold label questions ask about
"""

import json
import re
from typing import Dict, List, Any, Tuple


def identify_question_targets(question: str) -> Tuple[str, List[str]]:
    """
    Identify which level and fields a question is asking about.
    
    Args:
        question: A single investigative question
        
    Returns:
        Tuple of (level, list of field names)
    """
    question_lower = question.lower()
    
    # Determine level
    if "scene-level" in question_lower or "scene level" in question_lower:
        level = "Scene-Level"
    elif "entity-level" in question_lower or "entity level" in question_lower:
        level = "Entity-Level"
    elif "action-level" in question_lower or "action level" in question_lower:
        level = "Action-Level"
    elif "semantic-level" in question_lower or "semantic level" in question_lower:
        level = "Semantic-Level"
    else:
        level = None
    
    # Identify specific fields being asked about
    fields = []
    
    # Scene-Level fields
    if any(word in question_lower for word in ["location", "address", "street", "building", "house number"]):
        fields.append("Location_Clues")
    if any(word in question_lower for word in ["environment", "setting", "weather", "lighting", "time of day"]):
        fields.append("Environment")
    if any(word in question_lower for word in ["scene change", "transition"]):
        fields.append("Scene_Changes")
    
    # Entity-Level fields
    if any(word in question_lower for word in ["description", "appearance", "clothing", "physical", "build", "height", "features"]):
        fields.append("Description")
    if any(word in question_lower for word in ["position", "location of", "where"]):
        fields.append("Position")
    if any(word in question_lower for word in ["role", "identity"]):
        fields.append("Role_if_Clear")
    if any(word in question_lower for word in ["object", "item", "weapon", "equipment", "holding"]):
        fields.append("Objects")
    
    # Action-Level fields
    if any(word in question_lower for word in ["action", "doing", "movement", "activity", "behavior"]):
        fields.extend(["Primary_Actions", "Secondary_Actions"])
    if any(word in question_lower for word in ["interaction", "engaging", "contact"]):
        fields.append("Interactions")
    
    # Semantic-Level fields
    if any(word in question_lower for word in ["intent", "purpose", "goal", "trying to"]):
        fields.append("Intent_if_Visible")
    if any(word in question_lower for word in ["emotion", "emotional state", "mood", "affect"]):
        fields.append("Emotional_State")
    if any(word in question_lower for word in ["audio", "sound", "speech", "voice", "command"]):
        fields.append("Notable_Audio")
    
    return level, fields


def mask_fields_from_questions(
    structured_data: Dict[str, Any],
    questions: str
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Mask fields in structured data based on what the gold label questions ask about.
    
    Args:
        structured_data: Parsed structured details dictionary
        questions: Gold label questions string
        
    Returns:
        Tuple of (masked_data, list of masked field paths)
    """
    masked_data = json.loads(json.dumps(structured_data))  # Deep copy
    masked_fields = []
    
    # Parse questions - they're numbered lists
    question_lines = [q.strip() for q in questions.split('\n') if q.strip() and not q.strip().startswith('Here are')]
    
    # Extract individual gold questions 
    individual_questions = []
    for line in question_lines:
        if re.match(r'^\d+[\.\)\-\:]', line):
            individual_questions.append(line)
    
    # For each question, identify what it's asking about and mask those said fields
    for question in individual_questions:
        level, fields = identify_question_targets(question)
        
        if not level or not fields:
            continue
        
        if level not in masked_data:
            continue
        
        # Mask the identified fields
        for field in fields:
            if field in masked_data[level]:
                # Handle different data types
                value = masked_data[level][field]
                
                if isinstance(value, str) and value.strip():
                    masked_data[level][field] = "[MASK]"
                    masked_fields.append(f"{level}.{field}")
                elif isinstance(value, list) and len(value) > 0:
                    masked_data[level][field] = ["[MASK]"]
                    masked_fields.append(f"{level}.{field}")
                elif isinstance(value, dict):
                    # For nested dicts like People
                    masked_data[level][field] = {k: "[MASK]" for k in value.keys()}
                    masked_fields.append(f"{level}.{field}")
            
            # Special handling for People/Objects in Entity-Level
            if level == "Entity-Level" and field in ["Description", "Position", "Role_if_Clear"]:
                if "People" in masked_data[level] and isinstance(masked_data[level]["People"], list):
                    for i, person in enumerate(masked_data[level]["People"]):
                        if isinstance(person, dict) and field in person:
                            if person[field].strip() or field == "Description":  # Always mask Description if asked
                                masked_data[level]["People"][i][field] = "[MASK]"
                                masked_fields.append(f"{level}.People[{i}].{field}")
    
    return masked_data, masked_fields


def create_masked_dataset(
    input_jsonl_path: str,
    output_jsonl_path: str
):
    """
    Process the entire data.jsonl file and create a masked version based on gold label questions.
    
    Args:
        input_jsonl_path: Path to input data.jsonl
        output_jsonl_path: Path to output masked data.jsonl
    """
    
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
            
            # Apply masking based on what the questions ask about
            masked_data, masked_fields = mask_fields_from_questions(
                structured_data,
                example['questions']
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
    
    args = parser.parse_args()
    
    create_masked_dataset(
        input_jsonl_path=args.input,
        output_jsonl_path=args.output
    )
