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
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cuda:1",
            trust_remote_code=True,
        )
    
    def build_conversation(self):
        # Proper chat template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This is a police bodycam video. Describe what happens in this video in detail, focus on actions, reponses, details about people and the surroundings. Be specific."},
                    {"type": "video"},
                ],
            },
        ]
        return conversation
    
    def build_qa_conversation(self, questions):
        # Proper chat template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f" This is a police bodycam video. You are given a set of questions, based on the video, answer these questions:\n {questions}"},
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
