#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=128g                    # allocate 128 GB of memory
#SBATCH -J "grpo_ft"        # name of the job
#SBATCH -o grpo_ft%j.out          # name of the output file
#SBATCH -e grpo_ft%j.err          # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 12:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:H100:2             # request 2 H100 GPUs

cd $SLURM_SUBMIT_DIR/../..

module load python/3.11.10
module load cuda/12.4.0/3mdaov5

python -m venv grpo_env
source grpo_env/bin/activate

pip install --upgrade pip

# Install dependencies
pip install vllm
pip install trl
pip install numpy
pip install peft
pip install torch
pip install transformers
pip install datasets
pip install tqdm

python -m fine_tuning.GDPO_ft.gdpo_ft