from google import genai
import os
from dotenv import load_dotenv

class gemini_model:
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        # Load API key from env
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No GEMINI_API_KEY found in .env file")
        
        # Initialize variables
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
    
    def generate_distillation_model_qs(self, structured_details):
        """
        Takes structured details extracted from bodycam video and prompt Gemini to generate questions
        """
        
        prompt = f"""
        You are an AI assistant aiding law enforcement analysts reviewing body-worn camera footage.

        Your task:
        - Based on the provided structured details, generate a list of investigative questions.
        - Every question must be something a human could answer by watching the video.
        - The goal is to guide analysts toward visual clues, context, behavior, or environment details that may matter.

        Rules for your output:
        - Write 1 meaningful question per detail element.
        - Do NOT repeat facts already stated — ask what is *unknown or unclear* visually.
        - Focus areas include: body language, environment, timeline, objects, threat indicators, interaction dynamics, or visual anomalies.
        - Use clear, concise, professional language.
        - Format the output as a numbered list.

        Structured information provided:
        {structured_details}
        """
        
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
    

    def eval(self, caption_text, ground_truth, evaluation_prompt_template):
        prompt = evaluation_prompt_template.format(
            caption_text=caption_text,
            ground_truth=ground_truth
        )
        
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
    
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

        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )