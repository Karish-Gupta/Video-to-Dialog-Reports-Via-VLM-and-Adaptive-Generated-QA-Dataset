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
                    "text": (
                        "You are a careful, objective analyst reviewing police body-worn camera footage. "
                        "Use both the video and the transcript. If video and transcript disagree, prioritize the video unless the video is unclear. "
                        "Do NOT provide internal chain-of-thought. Do not infer intent, motivation, or emotion unless directly supported by observable evidence. "
                        "Do not present overall conclusions anywhere except in the FINAL ANSWER section."
                    )
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"What is the context summary from this video and this transcript: \n{transcript}?"},
                        {"type": "video"},],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This footage generally depicts [one-sentence neutral summary]."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Can you provide the timeline of observations?"},
                        ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "[MM:SS] – Description of observable event\n[MM:SS] – Description of observable event"}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "List the key actions by each person."},
                        ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "[Person 1]: [MM:SS] – Action\n[Person 2]: [MM:SS] – Action"}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What are the evidence-based inferences?"},
                        ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Conclusion: <short inference>\n"
                        "Evidence: [MM:SS] observations\n"
                        "Confidence: [0-100]\n"
                        "Notes: alternative explanation"
                    )
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Identify de-escalation or compliance cues."},
                        ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "[MM:SS] – Observable cue of compliance/non-compliance"}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What are the limitations and uncertainties?"},
                        ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Limited visibility due to camera angle; audio unclear at [MM:SS]."}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Give me the final answer summary."},
                        ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The footage shows [neutral summary]. Key turning points occurred at [timestamps]. "
                        "Uncertainties remain due to [limitations]."
                    )
                }
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
