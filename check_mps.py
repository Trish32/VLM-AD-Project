"""Quick sanity check that PyTorch sees the Apple-Silicon MPS backend."""
import torch


def main() -> None:
    print("PyTorch version:", torch.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Selected device:", device)
    x = torch.randn((2, 2), device=device)
    print("Test tensor device:", x.device)


if __name__ == "__main__":
    main()
