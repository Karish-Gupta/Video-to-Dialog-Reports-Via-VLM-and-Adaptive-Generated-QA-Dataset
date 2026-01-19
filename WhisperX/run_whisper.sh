#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=64g                     # allocate 64 GB of memory
#SBATCH -J "whisperx_batch"           # name of the job
#SBATCH -o whisperx_%j.out            # name of the output file
#SBATCH -e whisperx_%j.err            # name of the error file
#SBATCH -p short                        # partition to submit to (GPU for large-v3 model)
#SBATCH -t 08:00:00                   # time limit of 8 hours for batch processing
#SBATCH --gres=gpu:1                  # request 1 GPU
#SBATCH --cpus-per-task=8             # request 8 CPU cores

cd $SLURM_SUBMIT_DIR

module load python/3.11.10
module load ffmpeg/6.1.1
module load cuda/12.8.0

# Export CUDA library path explicitly
export LD_LIBRARY_PATH=/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/cuda-12.8.0-4fdo42oeq4exuxktvxhctgvdpvphvclf/lib64:$LD_LIBRARY_PATH

# Activate the environment
source ./whisper_env/bin/activate

# Set PyTorch to allow loading non-weights-only pickles (needed for VAD model)
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

# Set HuggingFace token for speaker diarization
export HF_TOKEN="${HF_TOKEN}"

echo "Starting batch processing of 359 videos in batches of 50..."
echo "Started at: $(date)"

# Batch 1: Videos 0-49
echo "Processing Batch 1: Videos 0-49"
python transcribe.py --start_index 0 --end_index 50

# Batch 2: Videos 50-99
echo "Processing Batch 2: Videos 50-99"
python transcribe.py --start_index 50 --end_index 100

# Batch 3: Videos 100-149
echo "Processing Batch 3: Videos 100-149"
python transcribe.py --start_index 100 --end_index 150

# Batch 4: Videos 150-199
echo "Processing Batch 4: Videos 150-199"
python transcribe.py --start_index 150 --end_index 200

# Batch 5: Videos 200-249
echo "Processing Batch 5: Videos 200-249"
python transcribe.py --start_index 200 --end_index 250

# Batch 6: Videos 250-299
echo "Processing Batch 6: Videos 250-299"
python transcribe.py --start_index 250 --end_index 300

# Batch 7: Videos 300-349
echo "Processing Batch 7: Videos 300-349"
python transcribe.py --start_index 300 --end_index 350

# Batch 8: Videos 350-359
echo "Processing Batch 8: Videos 350-359"
python transcribe.py --start_index 350 --end_index 359

echo "All batches completed!"
echo "Finished at: $(date)"
