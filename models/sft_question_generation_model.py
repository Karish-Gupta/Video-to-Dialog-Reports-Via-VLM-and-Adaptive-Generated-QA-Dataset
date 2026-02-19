import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from models.llm import *


class QuestionGenerationModelSFT(llm):
    def __init__(self, model_name, adapter_dir):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(base, adapter_dir, torch_dtype=torch.float16)
        self.model.eval()

        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.tokenizer = tokenizer

    def generate_questions(self, vlm_summary: str, structured_details: str) -> str:
        """Generate exactly four numbered investigative questions from structured details."""
        stop_marker = "<END_OF_QUESTIONS>"
        prompt = f"""You are an AI assistant aiding law enforcement analysts reviewing body-worn camera footage.

Your task: Based on the provided structured details, generate exactly 4 investigative questions.

Rules for your output:
- Write exactly 4 numbered questions (1.-4.) formatted as a numbered list.
- Do NOT repeat facts already stated.
- Use clear, concise, professional language.

Structured information provided:
{structured_details}

End the output by placing the following stop marker on its own line after question 4:
{stop_marker}

Generated questions:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        num_input_tokens = inputs["input_ids"].shape[1]

        stop_ids = self.tokenizer.encode(stop_marker, add_special_tokens=False)

        class StopOnSequence(StoppingCriteria):
            def __init__(self, stop_ids):
                self.stop_ids = stop_ids
            def __call__(self, input_ids, scores, **kwargs):
                seq = input_ids[0].tolist()
                n = len(self.stop_ids)
                if n == 0 or len(seq) < n:
                    return False
                return seq[-n:] == self.stop_ids

        stopping_criteria = StoppingCriteriaList([StopOnSequence(stop_ids)])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
            )

        gen_tokens = outputs[0][num_input_tokens:]
        generated_raw = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        if stop_marker in generated_raw:
            generated = generated_raw.split(stop_marker)[0].strip()
        else:
            generated = generated_raw.strip()

        m = re.search(r"(1\.[\s\S]*?4\.[\s\S]*?)(?=\n\s*\d+\.|\Z)", generated)
        if m:
            return m.group(1).strip()

        lines = [l for l in generated.splitlines() if re.match(r"^\s*\d+\.", l)]
        if lines:
            numbered = []
            for l in lines:
                if len(numbered) >= 4:
                    break
                numbered.append(l.strip())
            return "\n".join(numbered)

        return generated