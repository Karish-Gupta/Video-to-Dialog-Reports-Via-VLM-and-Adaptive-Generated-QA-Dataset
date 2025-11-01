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
    trust_remote_code=True,
    use_fast=True

)

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cuda:0",
    trust_remote_code=True,
)

# Load & sample video frames
video_1_path = "VLM/videos/high_way_bodycam_30_sec.mp4"
video_2_path = "VLM/videos/highway_bodycam_one_min.mp4"
video_3_path = "VLM/videos/highway_bodycam.mp4"
videos = [video_1_path, video_2_path, video_3_path]


frames_list = [8, 16, 32, 64, 128, 256]

# Proper chat template
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "This is a police bodycam video. Describe what happens in this video in detail."},
            {"type": "video"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Run inference on various framerates

for video_path in videos:
    
    print(70 * "=")
    print(f"Video: {video_path}")

    for n in frames_list:
        
        # Initialize frames
        vr = VideoReader(video_path, ctx=cpu())
        num_frames = min(len(vr), n)
        indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
        frames = [vr[i].asnumpy() for i in indices]

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
            do_sample=False
        )

        # Decode
        response = processor.batch_decode(output, skip_special_tokens=True)[0]
        print(f"Loaded {len(frames)} frames:", frames[0].shape)
        print("\nModel output:\n", response)
        print(70 * "=")
        torch.cuda.empty_cache()
