from llm import *
from transcript_context import *
import json

# Step 1 prompt
step_1_prompt = f"""
Given a police bodycam video transcript, extract key information into the following structured format.

Transcript:
{json.dumps(transcript_up_2_40)}

Output Format:
{{
   "Scene Observations": "Describe the environment, setting, and notable scene details.",
   "Action": "Summarize the actions conducted by all individuals involved.",
   "Intents/Reason": "Explain the possible reasons or motivations behind these actions.",
   "Response": "Describe how individuals react to the situation.",
   "Inference": "Provide a brief assessment or interpretation of the situation.",
   "Individuals Involved": "List all identifiable individuals, including names or roles (e.g., 'Officer', 'Suspect', 'Witness')."
}}

Respond ONLY with valid structured elements in the exact format shown above.
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