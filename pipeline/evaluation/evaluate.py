from utils.llm import *
from utils.gemini_llm import *
from ground_truths import copa_video_ground_truths, evaluation_rubric
import os 

OUTPUT_DIR = "VLM/output_results_whisper" # Folder with each video caption output
RESULTS_FILE = "evaluation_results.json"

# Initialize model
llm_model = "meta-llama/Llama-3.3-70B-Instruct"
llm_ = llm(llm_model)
gemini = gemini_model()


# Evaluation prompt template
evaluation_prompt_template = f"""
You are evaluating an generated caption from a police bodycam video.

Use the rubric below and give only a JSON response with numerical scores and a short justification.

Rubric:
{evaluation_rubric}

Ground Truths:
{copa_video_ground_truths}



Return output in the following JSON format:
{{
  "Factual Accuracy": <0-5>,
  "Completeness": <0-5>,
  "Visual Enrichment": <0-5>,
  "Clarity": <0-5>,
  "Justification": "<2-4 sentence explanation>"
}}
"""

def evaluate_caption(caption_text, ground_truth):
    response = gemini.eval(caption_text, ground_truth, evaluation_prompt_template)
    
    try:
        parsed = json.loads(response)
        return parsed
    
    except json.JSONDecodeError:
        print("LLM returned non-JSON output")
        return {"raw": response}

def run_evaluation():
    results = {}

    for filename in os.listdir(OUTPUT_DIR):
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, "r") as f:
            output_text = f.read().strip()
            #TODO Extract caption from QA Generated Caption


def calculate_score(factual_accuracy, completeness, visual_enrichment, clarity):
    percentage_score = ((factual_accuracy + completeness + visual_enrichment + clarity) / 20) * 100
    
    return {
        "Factual Accuracy": factual_accuracy, 
        "Completeness": completeness,
        "Visual Enrichment": visual_enrichment,
        "Clarity": clarity,
        "Total Score": percentage_score
    }
    
