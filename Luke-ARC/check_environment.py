"""Report the PVNet HTC software stack and verify that the allocated GPU works."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys


REQUIRED_DISTRIBUTIONS = (
    "PVNet",
    "torch",
    "lightning",
    "ocf-data-sampler",
    "cfgrib",
    "eccodes",
    "gcsfs",
    "ocf-blosc2",
    "xarray",
    "zarr",
    "numcodecs",
    "ipykernel",
)


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    missing = []
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
            print(f"{distribution}: MISSING")
        else:
            print(f"{distribution}: {version}")

    import torch

    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"PyTorch HIP build: {torch.version.hip}")
    print(f"CUDA-compatible GPU visible: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No GPU is visible to this job. Confirm it was submitted with --gres=gpu:1 and inspect the PyTorch/driver setup.")
        return 1

    device = torch.device("cuda:0")
    result = (torch.ones(2, device=device) + 1).sum().item()
    print(f"Allocated device: {torch.cuda.get_device_name(0)}")
    print(f"GPU tensor operation result: {result}")

    for module_name in ("ocf_data_sampler", "cfgrib", "eccodes", "gcsfs", "ocf_blosc2"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # Report binary/codec import errors in the job log.
            missing.append(f"{module_name} import failed: {type(exc).__name__}: {exc}")
            print(missing[-1])

    if missing:
        print("Environment preflight found missing packages or import failures.")
        return 1
    print("Environment and GPU preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
