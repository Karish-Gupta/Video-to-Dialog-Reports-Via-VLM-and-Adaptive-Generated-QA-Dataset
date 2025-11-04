import torch
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from decord import VideoReader, cpu
import numpy as np

from transcript_context import transcript_up_2_40, full_transcript
from qa_generation import *

# Quantization setup
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Load processor and model
model_name = "llava-hf/LLaVA-NeXT-Video-7B-hf"

processor = LlavaNextVideoProcessor.from_pretrained(
    model_name, 
    trust_remote_code=True,
    use_fast=True

)

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_name,
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

# Use LLM for chat summary
llm_model = "meta-llama/Meta-Llama-3-8B"
llm_ = llm(llm_model)

llm_prompt_transcript_2_40 = llm_.build_transcript_context(transcript_up_2_40)
llm_prompt_full = llm_.build_transcript_context(full_transcript)

summarized_transcript_2_40 = llm_.invoke(llm_prompt_transcript_2_40)
summarized_transcript_full = llm_.invoke(llm_prompt_full)

# Proper chat template
conversation_up_to_2_40 = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"This is a police bodycam video. Describe what happens in this video in detail. Transcript up to this point: {summarized_transcript_2_40}"},
            {"type": "video"},
        ],
    },
]

conversation_full = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"This is a police bodycam video. Describe what happens in this video in detail. Transcript up to this point: {summarized_transcript_full}"},
            {"type": "video"},
        ],
    },
]
prompt_up_to_2_40 = processor.apply_chat_template(conversation_up_to_2_40, add_generation_prompt=True)
prompt_full = processor.apply_chat_template(conversation_full, add_generation_prompt=True)

# Run inference on various framerates
i = 0

for video_path in videos:
    
    print(70 * "=")
    print(f"Video: {video_path}")

    for n in frames_list:
        
        # Initialize frames
        vr = VideoReader(video_path, ctx=cpu())
        num_frames = min(len(vr), n)
        indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
        frames = [vr[i].asnumpy() for i in indices]

        if i == 2:
            # Preprocess multimodal input
            inputs = processor(
                text=[prompt_full],
                videos=[frames],
                padding=True,
                return_tensors="pt"
            ).to(model.device)

        else: 
            # Preprocess multimodal input
            inputs = processor(
                text=[prompt_up_to_2_40],
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
    
    i += 1

