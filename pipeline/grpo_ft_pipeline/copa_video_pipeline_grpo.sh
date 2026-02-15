#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=128g                     # allocate 128 GB of memory
#SBATCH -J "copa_video_pipeline"              # name of the job
#SBATCH -o copa_video_pipeline%j.out         # name of the output file
#SBATCH -e copa_video_pipeline%j.err         # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 20:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:H200:1             # request 1 H200 GPU

cd $SLURM_SUBMIT_DIR/../..

module load python/3.11.10
module load cuda/12.4.0/3mdaov5

python -m venv grpo_env
source grpo_env/bin/activate

pip install --upgrade pip

# Install dependencies
pip install unsloth
pip install google-genai
pip install python-dotenv
pip install accelerate
pip install tensorboard
pip install trl
pip install numpy
pip install bitsandbytes
pip install peft
pip install torch
pip install transformers
pip install datasets
pip install tqdm


python -m pipeline.baseline_pipeline.copa_video_pipeline_grpo