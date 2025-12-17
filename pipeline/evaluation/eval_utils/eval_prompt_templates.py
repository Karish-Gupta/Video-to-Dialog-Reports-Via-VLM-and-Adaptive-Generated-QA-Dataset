
  # ┌──────────────────────────────┐
  # │            RUBRIC            │
  # └──────────────────────────────┘
factual_accuracy_rubric = """Factual Accuracy (0-5)
Measures whether statements in the caption are true given the ground truth. 
Check sentence by sentence with the ground truth to verify accuracy of the presented information.
The final score should be the number of statements grounded in truth divided by the total number of statements multiplied by 5.  
(e.g., if 4 out of 5 sentences are correct, the score would be 4/5 * 5 = 4)                                        
"""

coverage_completeness_rubric = """Coverage & Completeness (0-5)
Does the caption capture all relevant and important events?
Check if all events in the Important Details section of the ground truth are missing in the caption, with the final score reflecting the proportion of events included.
(e.g., if 4 out of 5 key events are included, the score would be 4/5 * 5 = 4)
Anything missing from Auxillary Details should not impact the score.
"""

visual_enrichment_rubric = """Visual Enrichment (Non-Transcript Information) (0-5)
Measures how well the caption adds useful visual details that are not present in the transcript.

| Score | Criteria                                                         |
| ----- | ---------------------------------------------------------------- |
| **5** | Adds highly relevant visual context that improves understanding. |
| **4** | Adds useful visual details but limited variety.                  |
| **3** | Some visual context, but sparse.                                 |
| **2** | Minimal relevant visual additions.                               |
| **1** | Tries to add visual info but inaccurate or irrelevant.           |
| **0** | No visual context added.                                         |
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
  {coverage_completeness_rubric}

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


