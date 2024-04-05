"""
Utility functions
"""

import torch

def get_torch_device():
    """
    ## Returns
    `torch.device`: cuda or mps device if available, else cpu
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
