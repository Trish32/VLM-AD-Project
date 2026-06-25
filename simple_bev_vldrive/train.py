import random

import numpy as np
import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_device() -> torch.device:
    return DEVICE


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "mps":
        torch.manual_seed(seed)


def verify_environment() -> None:
    print("PyTorch version:", torch.__version__)
    print("Device selected:", DEVICE)
    print("MPS available:", torch.backends.mps.is_available())
    print("MPS built:", torch.backends.mps.is_built())


if __name__ == "__main__":
    set_seed()
    verify_environment()
    print("Training environment is configured for Simple-BEV on Apple Silicon.")
