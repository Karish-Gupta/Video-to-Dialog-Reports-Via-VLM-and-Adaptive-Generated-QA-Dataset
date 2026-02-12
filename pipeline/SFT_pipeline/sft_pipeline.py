import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"  # same model used when training
ADAPTER_DIR = "./qwen3-30b-instruct-police-questions-lora-gemini-vlm"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
) 

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,   # omit this if you didn't use bitsandbytes
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base, ADAPTER_DIR, torch_dtype=torch.float16)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

prompt = "Structured information provided:\n<your structured_details here>\n\nGenerate 4 investigative questions:"

inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)