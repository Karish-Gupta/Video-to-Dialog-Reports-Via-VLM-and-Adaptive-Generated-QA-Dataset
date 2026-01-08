
  # ┌──────────────────────────────┐
  # │            RUBRIC            │
  # └──────────────────────────────┘
factual_accuracy_rubric = """Factual Accuracy (0-5)
Measures whether statements in the caption are true given the ground truth.

Rules:
- If the general idea is mentioned in the ground truth (any section), it is considered true. 
- If extra details are added that contradict the ground truth, penalize the score.
- Check sentence by sentence with the ground truth to verify accuracy.
- The final score should be the number of statements grounded in truth divided by the total number of statements multiplied by 5.
- (e.g., if 4 out of 5 sentences are correct, the score would be 4/5 * 5 = 4)
- Round to the nearest whole number.
"""

completeness_rubric = """Completeness (0-5)
Does the caption capture all relevant and important events listed in 'important_details'?

Rules:
- Retrieve the list of events from the 'important_details' section of the ground truth.
- Check if each event from that list is present in the generated caption.
- The final score is the number of included events divided by the total number of 'important_details' events, multiplied by 5.
- (e.g., if the JSON lists 5 important details and the caption includes 3 of them, the score is 3/5 * 5 = 3)
- Round to the nearest whole number.
"""

visual_enrichment_rubric = """Visual Enrichment (0-5)
Measures the proportion of specific visual details captured in the caption.

Rules:
- Retrieve the list of items from the 'visual_enrichment_details' section of the ground truth.
- Check if each specific visual detail from that list is described in the generated caption.
- The final score is the number of included visual details divided by the total number of 'visual_enrichment_details' items, multiplied by 5.
- (e.g., if the JSON lists 8 visual details and the caption includes 4 of them, the score is 4/8 * 5 = 2.5 -> Round to 3)
- Round to the nearest whole number.
"""


# ┌───────────────────────────────────────────────┐
# │         EVALUATION PROMPT TEMPLATES           │
# └───────────────────────────────────────────────┘
def evaluation_prompt_template_factual(caption, ground_truth):
  factual_accuracy_prompt = f"""
  You are evaluating an generated caption from a police bodycam video.

  Use the rubric below and give only a JSON response with numerical scores and a short justification.

  Rubric:
  {factual_accuracy_rubric}

  Ground Truth:
  {ground_truth}

  Generated caption:
  {caption}

  Return output in the following JSON format:
  {{
    "Factual Accuracy": <0-5>,
    "Justification": "<For each sentence in the caption, indicate whether it is 'True' or 'False' based on the ground truth. Provide a brief explanation for each assessment.>"
  }}
  """
  return factual_accuracy_prompt


def evaluation_prompt_template_complete(caption, ground_truth): 
  completeness_prompt = f"""
  You are evaluating an generated caption from a police bodycam video.

  Use the rubric below and give only a JSON response with numerical scores and a short justification.

  Rubric:
  {completeness_rubric}

  Ground Truth:
  {ground_truth}

  Generated caption:
  {caption}

  Return output in the following JSON format:
  {{
    "Completeness": <0-5>,
    "Justification": "<2-4 sentence explanation>"
  }}
  """
  return completeness_prompt


def evaluation_prompt_template_enrich(caption, ground_truth):
  visual_enrichment_prompt = f"""
  You are evaluating an generated caption from a police bodycam video.

  Use the rubric below and give only a JSON response with numerical scores and a short justification.

  Rubric:
  {visual_enrichment_rubric}

  Ground Truth:
  {ground_truth}

  Generated caption:
  {caption}

  Return output in the following JSON format:
  {{
    "Visual Enrichment": <0-5>,
    "Justification": "<2-4 sentence explanation>"
  }}
  """
  return visual_enrichment_prompt


