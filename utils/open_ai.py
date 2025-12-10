from openai import OpenAI
import os
import time
from dotenv import load_dotenv

class openai_model:
    def __init__(self, model_name: str = "gpt-4"):
        # Load API key from env
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No GEMINI_API_KEY found in .env file")
        
        # Initialize variables
        self.client = OpenAI()
        self.model = model_name
        
    def eval_safe(self, caption_text, ground_truth, evaluation_prompt_template):
        """
        Safely substitute caption and ground_truth into the template using simple
        .replace to avoid KeyError caused by stray braces in the template/rubrics.
        If the template is already formatted, this will leave it unchanged.
        """
        prompt = evaluation_prompt_template
        # prefer simple placeholder replacement to avoid str.format KeyError
        if "{caption}" in prompt or "{ground_truth}" in prompt:
            prompt = prompt.replace("{caption}", caption_text).replace("{ground_truth}", ground_truth)
        else:
            # handle templates that might use doubled braces or already be formatted
            prompt = prompt.replace("{{caption}}", caption_text).replace("{{ground_truth}}", ground_truth)

        return self.client.responses.create(
            model=self.model_name,
            input=prompt
        )


