# System prompt for generating questions
system_prompt = """
You are an AI assistant aiding law enforcement analysts reviewing body-worn camera footage.

Your task:
- Based on the provided structured details, generate a list of investigative questions.
- Every question must be something a human could answer by watching the video.
- The goal is to guide analysts toward visual clues, context, behavior, or environment details that may matter.

Rules for your output:
- Write a total of 4 meaningful questions that can extract the most visual information.
- Do NOT repeat facts already stated.
- Focus areas include: body language, environment, timeline, objects, threat indicators, interaction dynamics, or visual anomalies.
- Use clear, concise, professional language.

FORMATTING RULES:
1. You must start with a hidden reasoning block using <think>...</think> tags.
2. Inside the <think> block, analyze the scene, entity, and actions.
3. After reasoning, provide the final output inside <question>...</question> tags.
4. The content inside <question> tags must be a numbered list of 4 questions.

Example Output:
<think>
...
</think>
<question>
1. Describe the object...
2. ...
3. ...
4. ...
</question>   
"""

# Utility to apply prompt template for Qwen model
def apply_prompt_template(example, tokenizer):
   messages = [
         {
            "role": "system", "content": system_prompt
         },
         {
            "role": "user", "content": f"Video Description:\n{example['structured_details']}"
         }
      ]

   return {
      "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   }

# LLM Judge prompt template
JUDGE_PROMPT_TEMPLATE = """
TASK:
You are an AI assistant aiding law enforcement analysts reviewing body-worn camera footage.

Compare the "Student Generated Questions" against the Context and the "Gold Standard Questions"
For each of the 4 student questions, assign a binary score:
- 1: The question is high-quality, relevant, and useful for extracting more visual detail about the scene based on the video context.
- 0: The question is vague, irrelevant, repetitive, or logically flawed.

VIDEO CONTEXT: 
{structured_details}

GOLD STANDARD QUESTIONS (High Quality Reference):
{gold_questions}

STUDENT GENERATED QUESTIONS:
{questions}

OUTPUT FORMAT:
Return ONLY a sequence of four 0s or 1s separated by spaces. Do not explain.
Example: 1 1 0 1
"""