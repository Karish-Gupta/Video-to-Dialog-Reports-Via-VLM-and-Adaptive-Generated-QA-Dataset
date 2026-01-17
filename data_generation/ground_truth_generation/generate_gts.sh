#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=128g                     # allocate 128 GB of memory
#SBATCH -J "gt_generation"              # name of the job
#SBATCH -o gt_generation%j.out         # name of the output file
#SBATCH -e gt_generation%j.err         # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 1:00:00                   # time limit of 1 hour

cd $SLURM_SUBMIT_DIR/../..

module load python/3.10.2/mqmlxcf
module load cuda/12.4.0/3mdaov5

python -m venv env
source env/bin/activate

pip install --upgrade pip
pip install google-genai

python -m data_generation.ground_truth_generation.generate_gts