from google import genai
import os
import time
import json
from dotenv import load_dotenv

class GeminiModel:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        # Load API key from env
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No GEMINI_API_KEY found in .env file")
        
        # Initialize variables
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name   

    def eval(self, caption_text, ground_truth, evaluation_prompt_template):
        prompt = evaluation_prompt_template.format(
            caption_text=caption_text,
            ground_truth=ground_truth
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
    
    def invoke(self, prompt):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
    
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

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[my_file, prompt],
        )

        return response.text
    
    def generate_ground_truths(self, transcript, video_path):
        gt_prompt = f"""
        Generate the ground truths of what is observed in the video and transcript
        
        Important details: Major events that occur in the video that sum up the main idea of the clip
        Visual enrichment details: Any details that are not found in the transcript and can only be picked up visually 
        Auxiliary details: All other details
        
        Rules: 
        - RETURN STRICTLY VALID JSON ONLY.
        - DO NOT include markdown fences or explanations.
        - Ground truths should be in the following format extactly:

        {{  
            "important_details": ["...", "..."],
            "visual_enrichment_details": ["...", "..."],
            "auxiliary_details": ["...", "..."],
            "transcript": "<insert transcript text here as a JSON string>"
        }}    
        
        Transcript:
        {json.dumps(transcript)}
        """
        return self.vlm_invoke(video_path, gt_prompt)


    def generate_vlm_summary(self, video_path, transcript):
        prompt = f"""
        You are given a bodycam video transcript, and the video.
        Generate a caption that gives visual details about the video. 
        Include the following in caption: 

        - Describe the setting (Time of day, vehicles, buildings, etc.)
        - Objects in the frame (Weapons, items in hand, consumables, etc.)
        - Describe how items are being used (Is a weapon being fired, radio being held by officer, etc.)
        - Describe individuals (What are people wearing, color of vehicles, accessory items worn such as hats or glasses, etc.)
        - Actions each individual made (Officer stating instructions, civilians complying, etc.)

        Ensure captions are direct and formal.

        Write in active voice as much as possible.
        Be direct, concise, and concrete.
        Use direct quotes only when needed.
        Use a person's name if it is known.

        Transcipt:
        {transcript}
        """
        response = self.vlm_invoke(video_path, prompt)
        return response

    def generate_structured_details(self, vlm_summary):

        prompt = f"""
        You analyze police body-worn camera recordings.
        You will be given a visual summary of the video <summary>
        Your task is to extract factual key details strictly grounded in what can be:
        - Seen
        - Heard
        - Directly inferred from observable physical evidence
        Do NOT speculate or invent missing details.
        Return ONLY the JSON output structure—no commentary, no explanation.
        
        If a category has no evidence, return an empty string or empty list
    
        <summary>
        {vlm_summary}
        </summary>

        Extract and output key details using the following structure:

        {{
        "Scene-Level": {{
            "Environment": "",        // Indoors/outdoors, setting type, weather, lighting
            "Location_Clues": "",     // Visible signage, street names, inferred setting only if visually grounded
            "Scene_Changes": []       // Changes in environment or camera movement (entry/exit rooms, approach vehicle, etc.)
        }},
        
        "Entity-Level": {{
            "People": [
                {{
                "Description": "",     // Clothing, identifiers, notable appearance features
                "Role_if_Clear": "",   // officer, civilian, suspect
                "Position": ""         // relative spatial location (left/right/behind/near doorway/etc.)
                }}
            ],
            "Animals": [],
            "Objects": [
                {{
                "Type": "",
                "Location": "",
                "Attributes": ""       // visible characteristics: damaged, brand, color, shape
                }}
            ]
        }},

        "Action-Level": {{
            "Primary_Actions": [],     // major events or movements observed in order
            "Secondary_Actions": [],   // gestures, handling items, positioning, approach/retreat
            "Interactions": []         // human-object, human-human, object-object
        }},

        "Semantic-Level": {{
            "Intent_if_Visible": "",   // ONLY if visually clear (e.g., fleeing, surrendering, reaching for object)
            "Emotional_State": "",     // body language, tone indicators (NOT speculation)
            "Notable_Audio": []        // shouting, sirens, arguments, commands, unknown sounds
        }}
        }}
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
    


    def generate_questions(self, structured_output):
                
        prompt = f"""
        You are an AI assistant aiding law enforcement analysts reviewing body-worn camera footage.

        Your task:
        - Based on the provided structured details, generate a list of investigative questions.
        - Every question must be something a human could answer by watching the video.
        - The goal is to guide analysts toward visual clues, context, behavior, or environment details that may matter.

        Rules for your output:
        - Write a total of 4 meaningful questions that can extract the most visual information.
        - Do NOT repeat facts already stated.
        - Focus areas include: body language, environment, timeline, objects, threat indicators, interaction dynamics, or visual anomalies.
        - Use clear, concise, professional language.
        - Format the output as a numbered list.

        Structured information provided:
        {structured_output}
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
    
    def answer_questions(self, video_path, generated_qs):
        prompt = f"""
        This is a police bodycam video. You are given a set of questions, based on the video, answer these questions:\n {generated_qs}
        """
        response = self.vlm_invoke(video_path, prompt)
        return response
    

    def generate_qa_caption(self, vlm_summary, vlm_answers):
        prompt = f"""
        You are given a bodycam video transcript, visual summary, and question-answer pairs.
        Generate a caption that gives visual details about the video. 
        Ensure that you make use of the questions and answers to enhance the caption.
        Include the following in caption: 

        - Describe the setting (Time of day, vehicles, buildings, etc.)
        - Objects in the frame (Weapons, items in hand, consumables, etc.)
        - Describe how items are being used (Is a weapon being fired, radio being held by officer, etc.)
        - Describe individuals (What are people wearing, color of vehicles, accessory items worn such as hats or glasses, etc.)
        - Actions each individual made (Officer stating instructions, civilians complying, etc.)

        Write in active voice as much as possible.
        Be as detailed as possible.
        Use direct quotes only when needed.
        Use a person's name if it is known.

        Visual Summary:
        {vlm_summary}

        Answers:
        {vlm_answers}
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text