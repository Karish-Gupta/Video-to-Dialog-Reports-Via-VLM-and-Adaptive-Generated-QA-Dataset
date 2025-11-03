import torch

def get_device():
    """Get available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")