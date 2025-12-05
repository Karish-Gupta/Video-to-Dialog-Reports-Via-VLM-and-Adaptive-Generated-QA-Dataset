#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=128g                    # allocate 128 GB of memory
#SBATCH -J "approach2_ft"             # name of the job
#SBATCH -o approach2_ft%j.out         # name of the output file
#SBATCH -e approach2_ft%j.err         # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 12:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:H100:1             # request 1 H100 GPU

cd $SLURM_SUBMIT_DIR/..

module load python/3.11.10
module load cuda/12.4.0/3mdaov5

python -m venv env
source env/bin/activate

pip install --upgrade pip
pip install numpy
pip install peft
pip install torch
pip install transformers
pip install datasets
pip install tqdm
pip install scikit-learn
pip install accelerate

# First create the masked dataset
python -m Approach2.masking_utils \
    --input data.jsonl \
    --output Approach2/masked_data.jsonl \
    --mask_prob 0.3 \
    --min_masks 1 \
    --max_masks 5 \
    --seed 42

# Then train on it
python -m Approach2.train_masked_ft \
    --dataset Approach2/masked_data.jsonl \
    --output_dir approach2_finetuned_model \
    --train_size 100 \
    --eval_size 20 \
    --num_epochs 3
