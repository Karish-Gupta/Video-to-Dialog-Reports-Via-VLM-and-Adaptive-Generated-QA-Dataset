import torch
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from decord import VideoReader, cpu
import numpy as np

# Quantization setup
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Load processor and model
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

processor = LlavaNextVideoProcessor.from_pretrained(
    model_id, 
    trust_remote_code=True
)

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    trust_remote_code=True,
)

# Load & sample video frames
video_path = r"C:\Users\karis\WorcesterPolytechnicInstitute\MQP\VLM\videos\edited-storm-body-cam.mp4"

vr = VideoReader(video_path, ctx=cpu())
num_frames = min(len(vr), 32)
indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
frames = [vr[i].asnumpy() for i in indices]
print(f"Loaded {len(frames)} frames:", frames[0].shape)

# Proper chat template
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what happens in this video in detail."},
            {"type": "video"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Preprocess multimodal input
inputs = processor(
    text=[prompt],
    videos=[frames],
    padding=True,
    return_tensors="pt"
).to(model.device)

# Generate
output = model.generate(
    **inputs,
    max_new_tokens=250,
    temperature=0.2,
    do_sample=False
)

# Decode
response = processor.batch_decode(output, skip_special_tokens=True)[0]
print("\nModel output:\n", response)
