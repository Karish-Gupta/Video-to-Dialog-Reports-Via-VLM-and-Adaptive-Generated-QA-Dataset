from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# only the above imports are needed for qwen-32b-vl inference;
# other utilities were not used and have been removed


class Qwen32bVL:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-32B-Instruct"):
        # processor used to prepare text/video inputs
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        # apply optional 4-bit quantization for the large model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

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

# convenience global using default qwen-32b-vl
qwen_vl_instance = Qwen32bVL()

def qwen_vl_inference(video, prompt, **kwargs):
    return qwen_vl_instance.infer(video, prompt, **kwargs)

qwen_vl_instance.infer("path/to/video.mp4", "Describe the video in detail, focusing on actions, responses, details about people and the surroundings. Be specific.")