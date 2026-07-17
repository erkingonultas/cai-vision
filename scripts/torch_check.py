"""Quick CUDA availability check.

Prints whether PyTorch can see a CUDA-capable GPU, and which CUDA runtime
version it was built against. Useful to verify a CUDA install is wired up
before kicking off a long training run.
"""
import torch

from cai.device import is_cuda_available


def main() -> None:
    print(
        f"CUDA available: {is_cuda_available()}\n"
        f"PyTorch built with CUDA: {torch.version.cuda}"
    )


if __name__ == "__main__":
    main()