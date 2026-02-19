import json
import os
from openai import OpenAI

INPUT_JSONL = "distillation_results_gemini (1).jsonl"
OUTPUT_JSONL = "output_with_cot.jsonl"

def process_jsonl_file():
    client = OpenAI(
        api_key="",
        base_url="https://api.deepseek.com"
    )

    with open(INPUT_JSONL, "r", encoding="utf-8") as infile, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:

        for line_num, line in enumerate(infile, start=1):
            if not line.strip():
                continue

            entry = json.loads(line)

            prompt = f"""Analyze the following video description:

VIDEO INDEX: {entry['video_index']}

SUMMARY:
{entry['vlm_summary']}

STRUCTURED DETAILS:
{entry['structured_details']}

You are an AI assistant aiding law enforcement analysts reviewing content.

Your task:
- Based on the provided structured details, generate a list of investigative questions.
- Every question must be something a human could answer by examining the content.
- The goal is to guide analysts toward visual clues, context, behavior, or environment details.

Rules:
- Write exactly 4 questions.
- Do NOT repeat stated facts.
- Focus on body language, environment, timeline, objects, threat indicators, or interactions.
- Use professional, concise language.
- Output a numbered list only.
"""

            try:
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": "You are an expert video analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500,
                    extra_body={"thinking": {"type": "enabled"}}
                )

                msg = response.choices[0].message

                # DeepSeek thinking / CoT extraction (robust fallback)
                cot = response.choices[0].message.reasoning_content

                questions = msg.content

                output_entry = {
                    "index": entry["video_index"],
                    "CoT": cot,
                    "questions": questions
                }

                outfile.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

                print(f"[✓] Processed entry {entry['video_index']} (line {line_num})")
                

            except Exception as e:
                print(f"[✗] Error on line {line_num}: {e}")
                
        

if __name__ == "__main__":
    process_jsonl_file()
