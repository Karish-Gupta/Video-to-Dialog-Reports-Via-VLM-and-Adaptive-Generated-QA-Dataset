#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=64g                     # allocate 64 GB of memory 
#SBATCH -J "pipeline_vlm"             # name of the job
#SBATCH -o pipeline_%j.out            # name of the output file
#SBATCH -e pipeline_%j.err            # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 06:00:00                   # time limit of 6 hours
#SBATCH --cpus-per-task=8             # request 8 CPU cores
#SBATCH --gres=gpu:H100:1             # request 1 H100 GPU

cd $SLURM_SUBMIT_DIR

module load python/3.11.10
module load ffmpeg/6.1.1
module load cuda/12.1  # Load CUDA for GPU support

# Use existing environment or create new one
if [ ! -d "./whisper_env" ]; then
    python -m venv whisper_env
fi

# Activate the environment
source ./whisper_env/bin/activate

# Install/upgrade dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126


# Verify GPU is available
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Run the pipeline with GPU
echo "Starting pipeline..."
python main.py

echo "Pipeline complete! Check outputs2/ for results."

