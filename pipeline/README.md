This folder focuses on running the pipeline, as well as the evaluation script. Everything contained in Deprecated folder is non-essential and will be removed in future patches.

NEEDED TO RUN:
        Access to LLama-70B-Instruct and LlaVA-Next-Video-34B through huggingface, as well as an API token for hugging face.
        Tokens can be added to local enviornment using:
            export HF-TOKEN="<Your Hugging Face Token>"
        Or logging into hugging-face in terminal with the command:
            huggingface-cli login

    For evaluation, you will need a gemini API key in the .env folder (Check the readme in evaluation for more info)


RUNNING THE PIPELINE:
    The original pipeline runs through the copa_video_pipeline.py file. Assuming you have everything set up as mentioned in the Needed to Run section, you will be able to run the code through the copa_video_pipeline.sh file (you may have to make modifications to the specifics of the script depending on enviornment)

    If you want to follow the flow of execution, check utils/llm and utils/vlm for the functions and prompts being used

    The output of this file will be a txt file containing the outputs for each section

    The outputs for each video will be:
        VISUAL SUMMARY: An initial summary from the VLM
        STRUCTURED OUTPUT: A structured output containing key details from the transcript and visual summary
        GENERATED QUESTIONS: Questions generated based on the key details from the structured output
        QUESTION ANSWERED: The VLM answers to these questions
        VQA CAPTION: Using all this information, create a video caption
        NON-VQA CAPTION: Video caption with just the visual summary and transcript



    For help with evaluation, please check the readme in evaluation
    