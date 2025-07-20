import torch


def detect_device(verbose=True):
    if torch.cuda.is_available():
        device = "cuda"
        if verbose:
            print("Detected CUDA GPU. Using cuda.")
    # MPS detection for Apple Silicon (macOS)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = "mps"
        if verbose:
            print("Detected Apple Silicon (MPS). Using mps.")
    else:
        device = "cpu"
        if verbose:
            print("No GPU detected. Using cpu.")
    return device
