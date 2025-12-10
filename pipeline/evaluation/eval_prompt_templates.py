from pipeline.evaluation.ground_truths import copa_video_ground_truths, factual_accuracy_rubric, clarity_professionalism_rubric, coverage_completeness_rubric, visual_enrichment_rubric


evaluation_prompt_template_factual = f"""
You are evaluating an generated caption from a police bodycam video.

Use the rubric below and give only a JSON response with numerical scores and a short justification.

Rubric:
{factual_accuracy_rubric}

Ground Truth:
{{ground_truth}}

Generated caption:
{{caption}}

Return output in the following JSON format:
{{
  "Factual Accuracy": <0-5>,
  "Justification": "<2-4 sentence explanation>"
}}
"""

evaluation_prompt_template_complete = f"""
You are evaluating an generated caption from a police bodycam video.

Use the rubric below and give only a JSON response with numerical scores and a short justification.

Rubric:
{coverage_completeness_rubric}

Ground Truth:
{{ground_truth}}

Generated caption:
{{caption}}

Return output in the following JSON format:
{{
  "Completeness": <0-5>,
  "Justification": "<2-4 sentence explanation>"
}}
"""

evaluation_prompt_template_enrich = f"""
You are evaluating an generated caption from a police bodycam video.

Use the rubric below and give only a JSON response with numerical scores and a short justification.

Rubric:
{visual_enrichment_rubric}

Ground Truth:
{{ground_truth}}

Generated caption:
{{caption}}

Return output in the following JSON format:
{{
  "Visual Enrichment": <0-5>,
  "Justification": "<2-4 sentence explanation>"
}}
"""

evaluation_prompt_template_clarity = f"""
You are evaluating an generated caption from a police bodycam video.

Use the rubric below and give only a JSON response with numerical scores and a short justification.

Rubric:
{clarity_professionalism_rubric}

Ground Truth:
{{ground_truth}}

Generated caption:
{{caption}}

Return output in the following JSON format:
{{
  "Clarity": <0-5>,
  "Justification": "<2-4 sentence explanation>"
}}
"""
