import os
import json
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

class DeepSeekModel:
    def __init__(self, model_name: str = "deepseek-chat"):
        """Initialize DeepSeek model client."""
        # Load API key from env
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("No DEEPSEEK_API_KEY found in .env file")
        
        # Initialize OpenAI client for DeepSeek API
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model_name = model_name
        self.max_tokens = 4096  # Adjust based on model limits

    def invoke(self, 
               prompt: str, 
               system_message: Optional[str] = None,
               temperature: float = 0.1,
               json_mode: bool = False) -> str:
        """Basic text generation with optional system message."""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"} if json_mode else None,
            extra_body={"thinking": {"type": "enabled"}}
        )
        
        return response.choices[0].message.content

    def eval(self, 
             caption_text: str, 
             ground_truth: str, 
             evaluation_prompt_template: str) -> str:
        """Evaluate caption against ground truth."""
        prompt = evaluation_prompt_template.format(
            caption_text=caption_text,
            ground_truth=ground_truth
        )
        
        return self.invoke(prompt)

    def generate_structured_details(self, vlm_summary: str) -> Dict[str, Any]:
        """Generate structured details from visual summary."""
        prompt = f"""
        You analyze police body-worn camera recordings or similar content.
        You will be given a visual summary of the content <summary>
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
            "Scene_Changes": []       // Changes in environment or camera movement
        }},
        
        "Entity-Level": {{
            "People": [
                {{
                "Description": "",     // Clothing, identifiers, notable appearance features
                "Role_if_Clear": "",   // officer, civilian, suspect
                "Position": ""         // relative spatial location
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
            "Secondary_Actions": [],   // gestures, handling items, positioning
            "Interactions": []         // human-object, human-human, object-object
        }},

        "Semantic-Level": {{
            "Intent_if_Visible": "",   // ONLY if visually clear
            "Emotional_State": "",     // body language, tone indicators (NOT speculation)
            "Notable_Audio": []        // shouting, sirens, arguments, commands, unknown sounds
        }}
        }}
        """
        
        response = self.invoke(prompt, json_mode=True)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}

    def generate_questions(self, structured_output: Dict[str, Any]) -> str:
        """Generate investigative questions from structured details."""
        prompt = f"""
        You are an AI assistant aiding law enforcement analysts reviewing content.

        Your task:
        - Based on the provided structured details, generate a list of investigative questions.
        - Every question must be something a human could answer by examining the content.
        - The goal is to guide analysts toward visual clues, context, behavior, or environment details.

        Rules for your output:
        - Write a total of 4 meaningful questions that can extract the most visual information.
        - Do NOT repeat facts already stated.
        - Focus areas include: body language, environment, timeline, objects, threat indicators, interaction dynamics, or visual anomalies.
        - Use clear, concise, professional language.
        - Format the output as a numbered list.

        Structured information provided:
        {json.dumps(structured_output, indent=2)}
        """

        return self.invoke(prompt)


    def generate_with_retry(self, 
                           prompt: str, 
                           max_retries: int = 3, 
                           **kwargs) -> str:
        """Generate with retry logic for robustness."""
        for attempt in range(max_retries):
            try:
                return self.invoke(prompt, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"Attempt {attempt + 1} failed, retrying...")
                import time
                time.sleep(1)  # Brief delay before retry