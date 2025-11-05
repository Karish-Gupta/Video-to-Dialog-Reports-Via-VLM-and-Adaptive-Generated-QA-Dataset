from transcript_context import transcript_up_2_40, full_transcript
from qa_generation import *

# Use LLM for chat summary
llm_model = "meta-llama/Meta-Llama-3-70B"
llm_ = llm(llm_model)

llm_prompt_transcript_2_40 = llm_.build_transcript_context(transcript_up_2_40)
llm_prompt_full = llm_.build_transcript_context(full_transcript)

summarized_transcript_2_40 = llm_.invoke(llm_prompt_transcript_2_40)
summarized_transcript_full = llm_.invoke(llm_prompt_full)

print(f"Sumarized transcript 2_40: {summarized_transcript_2_40}")
print(f"Summarized transcript full: {summarized_transcript_full}")

