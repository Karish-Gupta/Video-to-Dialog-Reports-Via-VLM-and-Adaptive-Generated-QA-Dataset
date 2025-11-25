import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

if "HF_TOKEN" in os.environ:
   login(token=os.environ["HF_TOKEN"])

class FineTuner:
    # Getting started on preparing fine-tuning data and fine tuning the model.

    def __init__(self, model: str = "meta-llama/Meta-Llama-3-8B"):
        # Quantization config (4-bit)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )

        self.model = None
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                quantization_config=quant_config,
                device_map="cuda:0",
            )
        except Exception as e:
            print("Model load (4-bit) failed:", e)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
            except Exception as e2:
                print("Model load failed:", e2)

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


# ------------------------------------------------------------
# PEFT / LoRA Adapter (NOT IMPLEMENTED YET)
#
# Steps to do later:
#   1. Import PEFT:
#        from peft import LoraConfig, get_peft_model
#
#   2. Define a LoRA config targeting Q/K/V projections
#
#   3. Wrap the base model:
# ------------------------------------------------------------
    def scaffold_dataset_samples(self, out_path: str = ""):
       
        examples = [
            {
                "name": "example",
                "transcript": "[SAMPLE TRANSCRIPT HERE]",
                "vlm_summary": {"scene": "", "entities": [], "actions": []},
                "structured_outputs": {
                    "Scene-Level": {"Environment": "", "Location_Clues": "", "Scene_Changes": []},
                    "Entity-Level": {"People": [], "Animals": [], "Objects": []},
                    "Action-Level": {"Primary_Actions": [], "Secondary_Actions": [], "Interactions": []},
                    "Semantic-Level": {"Intent_if_Visible": "", "Emotional_State": "", "Notable_Audio": []},
                },
                "gold_questions": [
                    "What is the primary action observed in the scene?",
                    "Who are the visible people and what are they doing?",
                ],
            }
        ]

        d = os.path.dirname(out_path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as fh:
            for e in examples:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")

        print(f"Wrote scaffold samples to {out_path}")

    def convert_to_prompt_completion(self, example: dict, direction: str = "label->question"):
        """Convert an example to a text-generation prompt/completion pair."""
        if direction == "label->question":
            prompt = (
                "You are given a structured description of a video scene.\n"
                "Return one concise, high-quality question that would help clarify or expand understanding of the scene.\n\n"
                "Structured Output:\n"
                f"{json.dumps(example.get('structured_outputs', {}), ensure_ascii=False)}\n\n"
                "Question:"
            )
            completion = (example.get("gold_questions") or [""])[0]

        else:
            prompt = (
                "You are given a high-quality question about a video scene.\n"
                "Return a grounded structured JSON output strictly based on the question's focus.\n\n"
                "Question:\n"
                f"{(example.get('gold_questions') or ['WHAT_IS_SEEN?'])[0]}\n\n"
                "Structured Output (JSON):"
            )
            completion = json.dumps(example.get("structured_outputs", {}), ensure_ascii=False)
        return {"prompt": prompt, "completion": completion}

    def write_prompt_completion_jsonl(
        self,
        examples_path: str = "fine_tuning_samples.jsonl",
        out_path: str = "fine_tuning_prompt_completion.jsonl",
        direction: str = "label->question",
    ):
        """Convert all scaffolded examples to prompt-completion format."""
        if not os.path.exists(examples_path):
            raise FileNotFoundError(f"Examples file not found: {examples_path}")

        examples = []
        with open(examples_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    examples.append(json.loads(line))

        d = os.path.dirname(out_path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as fh:
            for ex in examples:
                pc = self.convert_to_prompt_completion(ex, direction=direction)
                fh.write(json.dumps(pc, ensure_ascii=False) + "\n")

        print(f"Wrote {len(examples)} prompt-completion examples to {out_path}")


if __name__ == "__main__":
    ft = FineTuner()
    ft.scaffold_dataset_samples()
    ft.write_prompt_completion_jsonl()
    
# ------------------------------------------------------------
# TRAINING LOOP (NOT IMPLEMENTED YET)
# ------------------------------------------------------------
# Accelerate Training Loop????
# ------------------------------------------------------------
#
#   from accelerate import Accelerator
#
#   accelerator = Accelerator()
#   model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
#
#   model.train()
#   for epoch in range(num_epochs):
#       for batch in dataloader:
#           outputs = model(**batch)
#           loss = outputs.loss
#           accelerator.backward(loss)
#           optimizer.step()
#           optimizer.zero_grad()
#
# ------------------------------------------------------------


