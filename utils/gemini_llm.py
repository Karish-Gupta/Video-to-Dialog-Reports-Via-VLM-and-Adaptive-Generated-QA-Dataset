from google import genai
import os
from dotenv import load_dotenv

class gemini_model:
    def __init__(self, model_name: str = "gemini-2.5-pro", temperature: float = 0.2):
        # Load API key from env
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No GEMINI_API_KEY found in .env file")
        
        # Initialize variables
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
    
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
            contents=prompt,
            generation_config={"temperature": self.temperature}
        )