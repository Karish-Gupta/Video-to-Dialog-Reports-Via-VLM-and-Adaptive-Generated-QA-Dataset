from models.gemini_model import GeminiModel
from models.deepseek_model import DeepSeekModel
from pipeline.evaluation.eval_utils.ground_truths import copa_video_ground_truths
import json
import re
import os
from typing import Any
from models.open_ai import *
from pipeline.evaluation.eval_utils.eval_prompt_templates import *
from pipeline.evaluation.eval_utils.calculate_averages import calculate_averages

# Initialize judges
gemini = GeminiModel()
deeppseek = DeepSeekModel()

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
    

def evaluate_caption(caption, ground_truth, model="DEEPSEEK"):
    prompts = {
        "Factual Accuracy": evaluation_prompt_template_factual(caption, ground_truth),
        "Completeness": evaluation_prompt_template_complete(caption, ground_truth),
        "Visual Enrichment": evaluation_prompt_template_enrich(caption, ground_truth),
    }
    results = {}
    for metric_name, prompt in prompts.items():
        model = model.upper()
        
        if model == "DEEPSEEK":
            resp = deeppseek.invoke(prompt)
        
        elif model == "GEMINI":
            resp = gemini.invoke(prompt)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        parsed = _extract_json_from_text(resp)
        results[metric_name] = parsed

    return results


def _extract_caption_from_output_file_QA(output_text: str) -> str:
    """
    Return only the content under the '=== QA CAPTION ===' header.
    """
    match = re.search(r"===\s*QA\s*CAPTION\s*===\s*(.*?)(?=\n===|\Z)", output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def _extract_caption_from_output_file_NQA(output_text: str) -> str:
    """
    Return only the content under the '=== QA CAPTION ===' header.
    """
    match = re.search(r"===\s*NON-QA\s*CAPTION\s*===\s*(.*?)(?=\n===|\Z)", output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def run_evaluation(OUTPUT_DIR="pipeline/output_results_whisper", RESULTS_FOLDER="pipeline/evaluation_results", NQA=False, QA=False, judge_model="DEEPSEEK"):
    """
    Run evaluation over all files in OUTPUT_DIR. For each flag that is True
    (NQA, QA), extract the corresponding caption section from the
    file, evaluate it, and save results to a separate JSON file under RESULTS_FOLDER.
    """
    # prepare output containers per-flag
    results_QA = {}
    results_NQA = {}

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    for filename in sorted(os.listdir(OUTPUT_DIR)):
        if not filename.lower().startswith("video"):
            continue

        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, "r", encoding="utf-8") as f:
            output_text = f.read()

        # Map video ID to ground truth, e.g., 'Video1_results.txt' -> 'video1'
        digits = re.findall(r"\d+", filename)
        video_key = f"video{digits[0]}" if digits else None
        ground_truth = copa_video_ground_truths.get(video_key)

        if not ground_truth:
            print(f"WARNING: No ground truth found for {filename}; skipping")
            continue

        # Small helper to evaluate and return structured result
        def _evaluate_and_score(caption_text: str):
            if not caption_text:
                return None
            eval_results = evaluate_caption(caption_text, ground_truth, judge_model)

            # Try to extract numeric values from parsed responses
            def _get_numeric(parsed, key_aliases):
                if not parsed:
                    return 0.0
                
                # Check specific aliases
                for alias in key_aliases:
                    if alias in parsed:
                        try:
                            return float(parsed[alias])
                        except (ValueError, TypeError):
                            continue
                
                # Fallback: scan values if specific keys failed
                for v in parsed.values() if isinstance(parsed, dict) else []:
                    if isinstance(v, (int, float)):
                        return float(v)
                    # Handle case where LLM returns string "4.5"
                    if isinstance(v, str):
                        try:
                            return float(v)
                        except ValueError:
                            continue
                return 0.0

            factual = _get_numeric(eval_results.get("Factual Accuracy", {}), ["Factual Accuracy", "Factual"])
            completeness = _get_numeric(eval_results.get("Completeness", {}), ["Completeness"])
            visual = _get_numeric(eval_results.get("Visual Enrichment", {}), ["Visual Enrichment", "Visual"])

            score = calculate_score(factual, completeness, visual)

            return {
                "file": filename,
                "caption": caption_text,
                "ground_truth": ground_truth,
                "raw_evaluation": eval_results,
                "score": score
            }

        # For each requested flag, extract corresponding caption and evaluate
        if QA:
            caption_text_qa = _extract_caption_from_output_file_QA(output_text)
            res = _evaluate_and_score(caption_text_qa)
            if res:
                results_QA[video_key] = res
                print(f"Evaluated QA {filename}: {res['score']}")

        if NQA:
            caption_text_nqa = _extract_caption_from_output_file_NQA(output_text)
            res = _evaluate_and_score(caption_text_nqa)
            if res:
                results_NQA[video_key] = res
                print(f"Evaluated NQA {filename}: {res['score']}")

    # Save results separately for each requested flag
    if QA:
        out_file = os.path.join(RESULTS_FOLDER, "evaluation_deepseek_QA_results.json")
        with open(out_file, "w", encoding="utf-8") as rf:
            json.dump(results_QA, rf, indent=2, ensure_ascii=False)
        print(f"Saved QA evaluation results: {out_file}")
        # Print averages for QA
        try:
            if results_QA:
                averages = calculate_averages(results_QA)
                print("\nAVERAGE SCORES (QA):")
                for category, avg in averages.items():
                    print(f"{category}: {avg:.2f}")
            else:
                print("No QA results to average.")
        except Exception as e:
            print(f"Could not compute QA averages: {e}")

    if NQA:
        out_file = os.path.join(RESULTS_FOLDER, "evaluation_deepseek_NQA_results.json")
        with open(out_file, "w", encoding="utf-8") as rf:
            json.dump(results_NQA, rf, indent=2, ensure_ascii=False)
        print(f"Saved NQA evaluation results: {out_file}")
        # Print averages for NQA
        try:
            if results_NQA:
                averages = calculate_averages(results_NQA)
                print("\nAVERAGE SCORES (NQA):")
                for category, avg in averages.items():
                    print(f"{category}: {avg:.2f}")
            else:
                print("No NQA results to average.")
        except Exception as e:
            print(f"Could not compute NQA averages: {e}")

def calculate_score(factual_accuracy, completeness, visual_enrichment):
    """
    Calculate aggregate score from three metrics (each 0-5). Total possible = 15.
    Returns a dict with the individual metric values and a percentage Total Score.
    """
    total = factual_accuracy + completeness + visual_enrichment
    percentage_score = (total / 15) * 100 if total is not None else 0

    return {
        "Factual Accuracy": factual_accuracy,
        "Completeness": completeness,
        "Visual Enrichment": visual_enrichment,
        "Total Score": percentage_score
    }

if __name__ == "__main__":
        
    output_dir = "pipeline/baseline_captions"
    results_folder = "pipeline/evaluation_results_baseline"
    judge_model = "DEEPSEEK"
    
    run_evaluation(OUTPUT_DIR=output_dir, RESULTS_FOLDER=results_folder, NQA=True, QA=True, judge_model=judge_model)


