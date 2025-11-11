from llm import *
from vlm import *
from transcript_context import *

# Initialize LLM
llm_model = "meta-llama/Llama-3.3-70B-Instruct"
llm_ = llm(llm_model)

vlm_model_name = "llava-hf/LLaVA-NeXT-Video-34B-hf"
vlm_ = vlm(vlm_model_name)

video_1_path = "VLM/videos/high_way_bodycam_30_sec.mp4"


# VLM summary
vlm_conversation = vlm_.build_conversation()
vlm_summary = vlm_.invoke(video_1_path, vlm_conversation)
print(f"VLM Summary:\n{vlm_summary}")

# Step 1 prompt
step_1_prompt = llm_.step_1_chat_template(transcript_60_sec, vlm_summary)
print(f"Step 1 Prompt:\n {step_1_prompt}")

structured_output = llm_.invoke(step_1_prompt)
print(f"Generated Structured Elements:\n {structured_output}")


# Step 2 prompt
step_2_prompt = llm_.step_2_chat_template(structured_output)
print(f"Step 2 Prompt:\n {step_2_prompt}")

generated_qs = llm_.invoke(step_2_prompt)
print(f"Generated Questions:\n {generated_qs}")

# Parse questions using regex
parsed_questions = llm_.parse_questions(generated_qs)
print(f"\n{'='*60}")
print(f"Parsed {len(parsed_questions)} questions:")
for i, q in enumerate(parsed_questions, 1):
    print(f"{i}. {q}")
print(f"{'='*60}\n")

# Pass each question one-by-one to VLM for answer generation
qa_pairs = []
for i, question in enumerate(parsed_questions, 1):
    print(f"\n{'='*60}")
    print(f"Processing Question {i}/{len(parsed_questions)}:")
    print(f"Q: {question}")
    print(f"{'='*60}")
    
    qa_conversation = vlm_.build_qa_conversation(question)
    vlm_answer = vlm_.invoke(video_1_path, qa_conversation)
    
    print(f"A: {vlm_answer}\n")
    
    qa_pairs.append({
        "question": question,
        "answer": vlm_answer
    })

# Print summary
print(f"\n{'='*60}")
print("FINAL QA PAIRS:")
print(f"{'='*60}")
for i, pair in enumerate(qa_pairs, 1):
    print(f"\nQ{i}: {pair['question']}")
    print(f"A{i}: {pair['answer']}")
print(f"{'='*60}\n") 
