#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=128g                     # allocate 128 GB of memory
#SBATCH -J "copa_video_pipeline"              # name of the job
#SBATCH -o copa_video_pipeline%j.out         # name of the output file
#SBATCH -e copa_video_pipeline%j.err         # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 20:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:H100:1             # request 1 H200 GPU

cd $SLURM_SUBMIT_DIR/../..

module load python/3.10.2/mqmlxcf
module load cuda/12.4.0/3mdaov5

python -m venv env
source env/bin/activate

pip install --upgrade pip
pip install --upgrade -q accelerate bitsandbytes
pip install transformers
pip install huggingface-hub
pip install -q av
pip install pillow
pip install torchvision
pip install protobuf
pip install sentencepiece
pip install torchcodec
pip install decord==0.6.0

python -m pipeline.copa_video_pipeline