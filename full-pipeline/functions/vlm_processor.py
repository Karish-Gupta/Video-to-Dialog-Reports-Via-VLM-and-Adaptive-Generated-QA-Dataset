import torch
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from typing import List, Dict
import os
from datetime import datetime

def load_llava_model(device: str = "cuda"):
    """
    Load LLaVA-NeXT-Video model for video understanding.
    """
    print(f"\n{'='*80}")
    print("Loading LLaVA-NeXT-Video model...")
    print(f"{'='*80}")
    
    model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    
    # Quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
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
    
    print("LLaVA-NeXT-Video model loaded successfully!")
    return model, processor

def process_chunk_with_vlm(frames: List, model, processor, 
                           prompt: str = "Describe what happens in this video segment in detail.") -> str:
    """
    Process video frames with LLaVA-NeXT-Video model.
    """
    if not frames:
        return "[No frames available for this chunk]"
    
    # Create conversation in chat template format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
            ],
        },
    ]
    
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Preprocess
    inputs = processor(
        text=[prompt_text],
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
    
    # Extract just the assistant's response (remove the prompt)
    if "assistant\n" in response:
        response = response.split("assistant\n")[-1].strip()
    
    return response

def save_vlm_descriptions(descriptions: List[Dict], output_dir: str = "outputs2", prefix: str = "vlm_descriptions"):
    """
    Save VLM descriptions to a timestamped text file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    txt_path = os.path.join(output_dir, f"{prefix}_{timestamp}.txt")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("VIDEO CHUNK DESCRIPTIONS (LLaVA-NeXT-Video)\n")
        f.write("="*80 + "\n\n")
        
        for desc in descriptions:
            chunk_id = desc['chunk_id']
            start_time = desc['start_time']
            end_time = desc['end_time']
            description = desc['description']
            
            # Format timestamps as HH:MM:SS
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d}"
            end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d}"
            
            f.write(f"Chunk {chunk_id} [{start_str} - {end_str}]\n")
            f.write("-" * 80 + "\n")
            f.write(f"{description}\n\n")
    
    print(f"VLM descriptions saved to: {txt_path}")
    return txt_path