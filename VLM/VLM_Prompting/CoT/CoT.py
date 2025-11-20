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
    
   
    def cot_prompting(self, transcript):
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": 
" You are a careful, objective analyst reviewing police body-worn camera footage. "
"Use both the video and the transcript. If video and transcript disagree, prioritize the video unless the video is unclear. "
"Do not infer intent, motivation, or emotion unless directly supported by observable evidence. "
"Do not present overall conclusions anywhere except in the FINAL ANSWER section. "

"1. TIMELINE OF OBSERVATIONS (FACTS ONLY, MAX 10 BULLETS)\n"
"List observable, timestamped events from both video and transcript.\n"

"2. EVIDENCE-BASED INFERENCES\n"
"Make reasoning based inferences based on the timeline of events with evidence:\n"

"3. FINAL ANSWER\n"
"Based on the above observations and inferences, provide a concise summary of the key events and findings from the footage. "
                }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":
    f"This is a police bodycam video. Using the video and the transcript below, produce the structured analysis described by the system instructions.\nTranscript:\n{transcript}"
                    },
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
            max_new_tokens=768,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=self.model.config.eos_token_id,
        )

        # Decode
        response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        print(f"Loaded {len(frames)} frames:", frames[0].shape)
        torch.cuda.empty_cache()
        return response
