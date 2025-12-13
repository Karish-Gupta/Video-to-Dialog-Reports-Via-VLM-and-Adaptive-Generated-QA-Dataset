from google import genai
import os
import time
from dotenv import load_dotenv

class gemini_model:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
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
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return response.candidates[0].content.parts[0].text
    

    def eval(self, caption_text, ground_truth, evaluation_prompt_template):
        prompt = evaluation_prompt_template.format(
            caption_text=caption_text,
            ground_truth=ground_truth
        )
        
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
    
    def invoke(self, prompt):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.candidates[0].content.parts[0].text
    
    def vlm_invoke(self, video_path, prompt):

        # Upload file
        my_file = self.client.files.upload(file=video_path)
        file_name = my_file.name
        
        # Poll until file becomes ACTIVE (or error)
        start = time.time()
        while True:
            meta = self.client.files.get(name=file_name)
            state = None
            if isinstance(meta, dict):
                state = meta.get("state") or meta.get("status")
            else:
                state = getattr(meta, "state", None) or getattr(meta, "status", None)

            if state == "ACTIVE":
                break
            if state in ("FAILED", "ERROR"):
                raise RuntimeError(f"File upload failed or rejected: {meta}")
            if time.time() - start > 100:
                raise TimeoutError(f"Timed out waiting for file to become ACTIVE (last state: {state})")
            time.sleep(1)

        return self.client.models.generate_content(
            model=self.model_name,
            contents=[my_file, prompt],
        )
    
    def step_2_chat_template(self, structured_output):
        prompt = f"""
         Based on the given structured information about a police bodycam video, generate specific questions based on key details:
      
         Questions to generate:
         1. Scene Observations  
         2. Items in Frame  
         3. Descriptions of Idividuals in Frame
         4. Actions 

         Examples:
         1. Why is the vehicle pulled over along the side of the road?
         2. What items are in the suspect's car?
         3. What is the age, ethnicity, and gender of the suspect?
         4. Why is the officer yelling profanity at the suspect?

         Structured Information:
        {structured_output}
      """
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
   