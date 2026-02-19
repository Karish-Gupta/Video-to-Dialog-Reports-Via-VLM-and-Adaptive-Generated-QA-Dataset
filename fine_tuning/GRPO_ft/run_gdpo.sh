#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=128g                    # allocate 128 GB of memory
#SBATCH -J "gdpo_ft"        # name of the job
#SBATCH -o gdpo_ft%j.out          # name of the output file
#SBATCH -e gdpo_ft%j.err          # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 12:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:H200:1             # request 1 H200 GPU

cd $SLURM_SUBMIT_DIR/../..

module load python/3.11.10
module load cuda/12.4.0/3mdaov5

python -m venv gdpo_env
source env/bin/activate

pip install --upgrade pip

# Install dependencies
pip install vllm==0.8.5.post1
pip install setuptools
pip install flash-attn --no-build-isolation

# Clone GDPO repo 
git clone https://github.com/NVlabs/GDPO.git
cd GDPO/trl-GDPO/trl-0.18.0-gdpo

# Install in editable mode
pip install -e .

pip install numpy
pip install bitsandbytes
pip install peft
pip install torch
pip install transformers
pip install datasets
pip install tqdm
pip install scikit-learn
pip install accelerate

python -m fine_tuning.GRPO_ft.grpo_ft