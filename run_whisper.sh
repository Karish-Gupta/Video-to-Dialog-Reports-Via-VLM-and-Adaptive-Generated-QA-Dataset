#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=32g                     # allocate 32 GB of memory
#SBATCH -J "whisperx_transcript"      # name of the job
#SBATCH -o whisperx_%j.out            # name of the output file
#SBATCH -e whisperx_%j.err            # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 04:00:00                   # time limit of 4 hours (CPU is slower)
#SBATCH --cpus-per-task=8             # request 8 CPU cores

cd $SLURM_SUBMIT_DIR

module load python/3.11.10
module load ffmpeg/6.1.1

# Remove incomplete environment and create fresh one
rm -rf ./whisper_env
python -m venv whisper_env

# Activate the environment
source ./whisper_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install whisperx yt-dlp torch
pip install -r requirements.txt

# Run the transcription script on CPU
python transribe.py

echo "Transcription completed!"
