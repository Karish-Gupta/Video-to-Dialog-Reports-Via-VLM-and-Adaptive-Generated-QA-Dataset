
  # ┌──────────────────────────────┐
  # │            RUBRIC            │
  # └──────────────────────────────┘
factual_accuracy_rubric = """Factual Accuracy (0-5)
Measures whether statements in the caption are true given the ground truth. 
Check sentece by sentence with the ground truth to verify accuracy of the presented information

| Score | Criteria                                                                                        |
| ----- | ----------------------------------------------------------------------------------------------- |
| **5** | All details are verifiably correct; no hallucinated events, objects, identities, or intentions. |
| **4** | Mostly correct with minor inaccuracies that do not change meaning.                              |
| **3** | Some inaccuracies or speculative language, but core meaning is correct.                         |
| **2** | Several incorrect statements or guesses; meaning is partially misleading.                       |
| **1** | Mostly inaccurate or speculative.                                                               |
| **0** | Completely fabricated or unrelated to the content.                                              |
"""

coverage_completeness_rubric = """Coverage & Completeness (0-5)
Does the caption capture all relevant and important events?

| Score | Criteria                                                                        |
| ----- | ------------------------------------------------------------------------------- |
| **5** | Captures all key actions, objects, people, environmental context, and sequence. |
| **4** | Covers most important details but misses a few.                                 |
| **3** | Adequate but incomplete.                                                        |
| **2** | Important information missing or unclear.                                       |
| **1** | Barely covers relevant content.                                                 |
| **0** | Provides no meaningful coverage.                                                |
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
    "Justification": "<2-4 sentence explanation>"
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


