# Instructions for WhisperX Transcription Code: 
This folder contains the WhisperX transcription pipeline used for generating high-quality transcripts from MP4 (or other audio/video) files on an HPC cluster using SLURM.
The pipeline runs a Python script (`transribe.py`) through an SLURM batch job and produces `.txt`, `.srt`, and `.json` transcription outputs.

## NEEDED TO RUN:
- Tokens can be added to local enviornment using:
```export HF-TOKEN="<Your Hugging Face Token>"```
Or logging into hugging-face in terminal with the command:
            ```huggingface-cli login```

To run this pipeline on your cluster, you must have:

### **Cluster Requirements**
- SLURM job scheduler  
- Access to the following cluster modules:
  - `python/3.11.10`
  - `ffmpeg/6.1.1`
- A compute node with:
  - **8 CPU cores**
  - **32 GB RAM**
  - **Up to 4 hours wall time** (configurable)

### **Hugging Face Access Requirements**
WhisperX and several of its dependent libraries use **Hugging Face models that require authentication**.
You must provide your Hugging Face token in one of the following ways:

#### **Option 1 — Add token to your environment**
```bash export HF_TOKEN="<Your Hugging Face Token>"```

### **Option 2 — CLI config**
```huggingface-cli login```
