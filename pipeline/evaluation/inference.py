from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
import warnings
import os 
import dashscope
import torch
import numpy as np
from decord import VideoReader, cpu
from PIL import Image
import math, hashlib, requests
from IPython.display import Markdown, display


class Qwen32bVL:
    """Lightweight wrapper for qwen-32b-vl multimodal inference."""
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-32B-Instruct"):
        # load processor and model just like the generic inference above
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    def infer(self,
              video,
              prompt,
              max_new_tokens: int = 2048,
              total_pixels: int = 20480 * 32 * 32,
              min_pixels: int = 64 * 32 * 32,
              max_frames: int = 2048,
              sample_fps: int = 2,
    ) -> str:
        """Run the same pipeline as :func:`inference` but with this model instance."""
        messages = [
            {"role": "user", "content": [
                    {"video": video,
                     "total_pixels": total_pixels,
                     "min_pixels": min_pixels,
                     "max_frames": max_frames,
                     "sample_fps": sample_fps},
                    {"type": "text", "text": prompt},
                ]
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages], return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt",
        )
        inputs = inputs.to('cuda')

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]

def qwen_vl_inference(video, prompt, **kwargs):
    return qwen_vl_instance.infer(video, prompt, **kwargs)


# convenience global using default qwen-32b-vl
qwen_vl_instance = Qwen32bVL()
qwen_vl_instance.qwen_vl_inference("path/to/video.mp4", "Describe the video in detail, focusing on actions, responses, details about people and the surroundings. Be specific.") 

