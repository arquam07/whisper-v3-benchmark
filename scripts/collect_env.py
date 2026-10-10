#!/usr/bin/env python3
"""Print the hardware and library versions to record alongside a benchmark run."""
import platform
import subprocess


def version(module: str) -> str:
    try:
        return __import__(module).__version__
    except Exception as exc:
        return f"not available ({type(exc).__name__})"


print(f"os            {platform.platform()}")
print(f"python        {platform.python_version()}")
for name in ("torch", "transformers", "av", "numpy", "fastapi"):
    print(f"{name:<13} {version(name)}")
try:
    import torch

    print(f"cuda (torch)  {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"gpu {i}         {torch.cuda.get_device_name(i)} ({total:.1f} GiB)")
except Exception as exc:
    print(f"torch/cuda    not available ({type(exc).__name__})")
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    print("nvidia-smi    " + out.stdout.strip().replace("\n", "\n              "))
except Exception as exc:
    print(f"nvidia-smi    not available ({type(exc).__name__})")
