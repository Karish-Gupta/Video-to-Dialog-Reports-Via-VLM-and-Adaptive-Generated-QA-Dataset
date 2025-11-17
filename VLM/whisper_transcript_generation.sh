#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=128g                     # allocate 128 GB of memory
#SBATCH -J "whisper_transcript_generation"              # name of the job
#SBATCH -o whisper_transcript_generation%j.out         # name of the output file
#SBATCH -e whisper_transcript_generation%j.err         # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 5:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:H100:1             

cd $SLURM_SUBMIT_DIR/..

module load python/3.11.10
module load cuda/12.4.0/3mdaov5
# module load cudnn8.9-cuda12.3/8.9.7.29
module load ffmpeg/6.1.1

python -m venv whisper_env
source ./whisper_env/bin/activate

pip install --upgrade pip
pip install transformers
pip install huggingface-hub
pip install whisperx 
pip install torch 
pip install ffmpeg-python
# pip install pyannote.audio
pip install opencv-python Pillow scipy


python VLM/whisper_transcript_generation.py