import torch
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from decord import VideoReader, cpu
import numpy as np
from huggingface_hub import login
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
   login(token=os.environ["HF_TOKEN"])
   
class vlm: 
    def __init__(self, model_name):
                
        # Quantization setup
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )


        self.processor = LlavaNextVideoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True,
            use_fast=True
        )

        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cuda:1",
            trust_remote_code=True,
        )
    
    def zero_shot_prompting(self, transcript):
        # Proper chat template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f" This is a police bodycam video. Based on the video and the following transcript, create a summary from this example: The bodycam shows the officer approaching a vehicle during a routine traffic stop for a broken taillight. The driver appears nervous, prompting the officer to use clear, calm verbal instructions to maintain safety. As the driver reaches for documents, the officer adjusts positioning to keep both hands visible. The footage highlights controlled communication and proper situational awareness during a low-risk encounter.\n {transcript}"},
                    {"type": "video"},
                ],
            },
        ]
        return conversation
    
    def invoke(self, video_path, conversation):
        # Process text conversation
        processed_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Extract frames from video
        vr = VideoReader(video_path, ctx=cpu())
        num_frames = min(len(vr), 128)
        indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
        frames = [vr[i].asnumpy() for i in indices]

        # Preprocess multimodal input
        inputs = self.processor(
            text=[processed_text],
            videos=[frames],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

        # Decode
        response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        print(f"Loaded {len(frames)} frames:", frames[0].shape)
        torch.cuda.empty_cache()
        return response
