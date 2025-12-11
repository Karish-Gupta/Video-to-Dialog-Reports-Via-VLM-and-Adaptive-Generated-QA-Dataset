from openai import OpenAI
import os
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
        
    def invoke(self, prompt):
        return self.client.responses.create(
            model=self.model,
            input=prompt
        )


