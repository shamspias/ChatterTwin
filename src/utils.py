import torch

def detect_device(verbose=True):
    if torch.cuda.is_available():
        device = "cuda"
        if verbose:
            print("Detected CUDA GPU. Using cuda.")
    elif getattr(torch, 'has_mps', False) and torch.has_mps:
        device = "mps"
        if verbose:
            print("Detected Apple Silicon. Using mps.")
    else:
        device = "cpu"
        if verbose:
            print("No GPU detected. Using cpu.")
    return device
