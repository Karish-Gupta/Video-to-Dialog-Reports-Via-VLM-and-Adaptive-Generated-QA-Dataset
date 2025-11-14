#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=128g                     # allocate 32 GB of memory
#SBATCH -J "VQApipeline"              # name of the job
#SBATCH -o VQA_%j.out            # name of the output file
#SBATCH -e VQA_%j.err            # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 04:00:00                   # time limit of 4 hours (CPU is slower)
#SBATCH --cpus-per-task=8             # request 8 CPU cores
#SBATCH --gres=gpu:H100:2             # request 1 H200 GPU

cd $SLURM_SUBMIT_DIR

module load python/3.11.10
module load ffmpeg/6.1.1
module load cuda/12.4.0/3mdaov5


# Remove incomplete environment and create fresh one
rm -rf ./whisper_env
python -m venv whisper_env

# Activate the environment
source ./whisper_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade -q accelerate bitsandbytes
pip install transformers
pip install huggingface-hub
pip install -q av
pip install pillow
pip install protobuf
pip install sentencepiece
pip install torchcodec
pip install decord==0.6.0

# Run the transcription script on CPU
python run_pipeline.py

echo "Transcription completed!"
