#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=128g                     # allocate 128 GB of memory
#SBATCH -J "whsiper_transcript_generation"              # name of the job
#SBATCH -o whsiper_transcript_generation%j.out         # name of the output file
#SBATCH -e whsiper_transcript_generation%j.err         # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 5:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:A100:1             

cd $SLURM_SUBMIT_DIR/..

module load python/3.10.2/mqmlxcf
module load cuda/12.4.0/3mdaov5

python -m venv env
source env/bin/activate

pip install --upgrade pip
pip install transformers
pip install huggingface-hub
pip install git+https://github.com/m-bain/whisperx.git
pip install pyannote.audio


python VLM/whsiper_transcript_generation.py