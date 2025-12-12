EVALUATION:

Evaluation on the data is performed from the evaluation.py file. This takes in a couple cli inputs, which are defined as follows:
    --nqa: Tells the user if they want to evaluate the Non - Question Answer part of the pipeline. Takes in either true or false as arguments, with the default being false
    --qa:  Tells the user if they want to evaluate the  Question Answer part of the pipeline. Takes in either true or false as arguments, with the default being false
    --summary: Tells the user if they want to evaluate the  Visual Summary part of the pipeline. Takes in either true or false as arguments, with the default being false
    --all: Tells the user if they want to evaluate all previous parts of the pipeline. Takes in either true or false as arguments, with the default being false
    --output-dir: The location of all outputs to be evaluated. set by default to pipeline/output_results_whisper
    --results-folder: The folder with all the evaluation results in json format. set by default to pipeline/evaluation_results

Example usage:
python pipeline/evaluation.py --nqa --qa

python pipeline/evaluation.py --all --output-dir "pipeline/to_be_evaluated" --results-folder "pipeline/evaluation_results"
