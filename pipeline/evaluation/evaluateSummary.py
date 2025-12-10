from utils.llm import *
from utils.gemini_llm import *
from pipeline.evaluation.ground_truths import copa_video_ground_truths, factual_accuracy_rubric, clarity_professionalism_rubric, coverage_completeness_rubric, visual_enrichment_rubric
import os 
import json
import re
from typing import Dict, Any
from utils.open_ai import *



OUTPUT_DIR = "pipeline/output_results_whisper" # Folder with each video caption output
VIDEO_DIR = "pipeline/copa_videos"  # Folder with each video file
RESULTS_FILE = "pipeline/evaluation_VS_results.json"


gemini = gemini_model()
gpt = openai_model()


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

def _extract_json_from_text(text: str) -> Any:
    # Try to find outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # fallback: try to clean curly quote issues
            candidate = candidate.replace("“", "\"").replace("”", "\"").replace("’", "'")
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return {"raw": text}
    else:
        # fallback: return raw output
        return {"raw": text}


def _gemini_text(resp) -> str:
    # Convert genai model response to a string.
    # The SDK response format can vary; falling back to str() if necessary
    try:
        # If using the new client, look for .candidates / .outputs
        if hasattr(resp, "candidates") and len(resp.candidates) > 0:
            c = resp.candidates[0]
            if hasattr(c, "content"):
                if isinstance(c.content, list):
                    # Sometimes content is a list of dicts with 'text'
                    texts = [x.get("text") if isinstance(x, dict) else str(x) for x in c.content]
                    return "\n".join(filter(None, texts))
                return str(c.content)
            if hasattr(c, "output"):
                return str(c.output)
        # Fallback:
        return str(resp)
    except Exception:
        return str(resp)


def evaluate_caption(caption_text, ground_truth):
    prompts = {
        "Factual Accuracy": evaluation_prompt_template_factual.replace("{caption}", caption_text).replace("{ground_truth}", ground_truth),
        "Completeness": evaluation_prompt_template_complete.replace("{caption}", caption_text).replace("{ground_truth}", ground_truth),
        "Visual Enrichment": evaluation_prompt_template_enrich.replace("{caption}", caption_text).replace("{ground_truth}", ground_truth),
        "Clarity": evaluation_prompt_template_clarity.replace("{caption}", caption_text).replace("{ground_truth}", ground_truth)
    }
    results = {}
    for metric_name, prompt_text in prompts.items():
        resp = gemini.eval_safe(caption_text, ground_truth, prompt_text)
        #raw_text = _gemini_text(resp) IF USING GEMINI
        raw_text = resp.output_text #IF OPENAI
        parsed = _extract_json_from_text(raw_text)
        results[metric_name] = parsed

    return results

def _extract_caption_from_output_file(output_text: str) -> str:
    """
    Return only the content under the '=== QA CAPTION ===' header.
    """
    match = re.search(r"===\s*VLM\s*SUMMARY\s*===\s*(.*?)(?=\n===|\Z)", output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def run_evaluation():
    results = {}

    for filename in sorted(os.listdir(OUTPUT_DIR)):
        if not filename.lower().startswith("video"):
            continue

        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, "r", encoding="utf-8") as f:
            output_text = f.read()
        caption_text = _extract_caption_from_output_file(output_text)

        # Map video ID to ground truth, e.g., 'Video1_results.txt' -> 'video1'
        digits = re.findall(r"\d+", filename)
        video_key = f"video{digits[0]}" if digits else None
        ground_truth = copa_video_ground_truths.get(video_key, "")

        if not ground_truth:
            print(f"WARNING: No ground truth found for {filename}; skipping")
            continue

        eval_results = evaluate_caption(caption_text, ground_truth)

        # Try to extract numeric values from parsed responses
        def _get_numeric(parsed, key_aliases):
            for alias in key_aliases:
                if alias in parsed and isinstance(parsed[alias], (int, float)):
                    return int(parsed[alias])
            # Some outputs may return different keys
            for v in parsed.values() if isinstance(parsed, dict) else []:
                if isinstance(v, (int, float)):
                    return int(v)
            return 0

        factual = _get_numeric(eval_results.get("Factual Accuracy", {}), ["Factual Accuracy", "Factual"])
        completeness = _get_numeric(eval_results.get("Completeness", {}), ["Completeness"])
        visual = _get_numeric(eval_results.get("Visual Enrichment", {}), ["Visual Enrichment", "Visual"])
        clarity = _get_numeric(eval_results.get("Clarity", {}), ["Clarity"])

        score = calculate_score(factual, completeness, visual, clarity)

        results[video_key] = {
            "file": filename,
            "caption": caption_text,
            "ground_truth": ground_truth,
            "raw_evaluation": eval_results,
            "score": score
        }
        print(f"Evaluated {filename}: {score}")

    # Save overall results
    with open(RESULTS_FILE, "w", encoding="utf-8") as rf:
        json.dump(results, rf, indent=2, ensure_ascii=False)
    print(f"Saved evaluation results: {RESULTS_FILE}")


def calculate_score(factual_accuracy, completeness, visual_enrichment, clarity):
    percentage_score = ((factual_accuracy + completeness + visual_enrichment + clarity) / 20) * 100
    
    return {
        "Factual Accuracy": factual_accuracy, 
        "Completeness": completeness,
        "Visual Enrichment": visual_enrichment,
        "Clarity": clarity,
        "Total Score": percentage_score
    }

run_evaluation()
    
