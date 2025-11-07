from llm import *
from transcript_context import *
import json

# Step 1 prompt
step_1_prompt = f"""
You are given a police bodycam transcript inside <transcript> tags.
Extract key details and return ONLY valid JSON.

<transcript>
{json.dumps(transcript_up_2_40)}
</transcript>

Output JSON structure:
{{
  "Scene Observations": "",
  "Action": "",
  "Intents/Reason": "",
  "Response": "",
  "Inference": "",
  "Individuals Involved": ""
}}
"""

# Initialize LLM
llm_model = "meta-llama/Meta-Llama-3-70B"
llm_ = llm(llm_model)

structured_output = llm_.invoke(step_1_prompt)
print(f"Generated Structured Elements: {structured_output}")


# Step 2 prompt
step_2_prompt = f"""
Based on the given structured information about a police bodycam video, generate thoughtful and specific questions based on pair combinations of each structured element of the video given below:

Structured information: 
{structured_output}

Questions to generate:
1. Scene Observation + Action
2. Scene Observation + Intent/Reason
3. Scene Observation + Response
4. Scene Observation + Inference
5. Scene Observation + Individuals Involved
6. Action + Intent/Reason
7. Action + Response
8. Action + Inference
9. Action + Individuals Involved
10. Intent/Reason + Response
11. Intent/Reason + Inference
12. Intent/Reason + Individuals Involved
13. Response + Inference
14. Response + Individuals Involved
15. Inference + Individuals Involved
"""

# Step 2 call LLM
generated_qs = llm_.invoke(step_2_prompt)
print(f"Generated Questions: {generated_qs}")