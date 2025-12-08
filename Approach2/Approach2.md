# Approach 2 Verification

## Summary
Approach 2 is a fine-tuning strategy that teaches the model to generate investigative questions about MISSING information, rather than just paraphrasing available information.

## How It Works

### Traditional Fine-Tuning (Baseline)
- **Input**: Complete structured details with all fields populated
- **Output**: Generate questions based on the complete information
- **Learning**: Model learns to paraphrase/reformat available information into questions

### Approach 2 (Our Method)
- **Input**: Masked structured details where fields mentioned in gold questions are replaced with `[MASK]`
- **Output**: Same questions (asking about the masked fields)
- **Learning**: Model learns to identify information gaps and generate questions to fill them

## Example

### Original Data (Complete)
```json
{
  "Scene-Level": {
    "Location_Clues": "Residential area with a fence or gate in the background"
  },
  "Entity-Level": {
    "People": [{"Description": "Person in dark clothing"}]
  }
}
```

### Approach 2 (Masked)
```json
{
  "Scene-Level": {
    "Location_Clues": "[MASK]"
  },
  "Entity-Level": {
    "People": [{"Description": "[MASK]"}]
  }
}
```

### Target Questions (Same for Both)
1. **Scene-Level:** Are there any visible street signs, house numbers, or unique features that could help identify the exact address?
2. **Entity-Level:** What specific descriptive details (clothing, build, height) can be identified?

## Why This Is Novel

1. **Information Gap Learning**: Model learns to recognize what's missing, not just what's present
2. **Robust to Incomplete Data**: Trained on scenarios where information is incomplete
3. **Focus on Investigation**: Questions target gaps in knowledge, mimicking real investigative workflow
4. **Not Just Noise**: Masking is intelligent - only fields that questions ask about are masked

## Masking Statistics
- Success Rate: ~63% of questions successfully matched to maskable fields
- Failsafe mechanism skips questions that can't be matched (avoiding noise)
- Parse errors handled gracefully