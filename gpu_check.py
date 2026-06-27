import os
import subprocess
import sys
from pathlib import Path


def _run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return str(e)


def check():
    print("=== GPU Diagnostic ===")
    print()

    print("--- System ---")
    print(f"platform: {sys.platform}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '(not set)')}")
    print(f"PATH: {os.environ.get('PATH', '(not set)')[:200]}...")
    print()

    print("--- CUDA / cuDNN libraries ---")
    for lib in ["libcuda", "libcudart", "libcublas", "libcublasLt", "libcufft",
                 "libcurand", "libcusolver", "libcusparse", "libcudnn", "libcudnn_adv",
                 "libcudnn_cnn", "libcudnn_ops"]:
        result = _run(f"ldconfig -p | grep {lib}")
        if lib in result and "not found" not in result.lower():
            lines = result.splitlines()
            print(f"  {lib}: found ({len(lines)} entries)")
            for line in lines[:2]:
                print(f"    {line.strip()}")
        else:
            print(f"  {lib}: MISSING")
    print()

    print("--- CUDA toolkit path ---")
    for d in ["/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-12.4"]:
        if Path(d).exists():
            contents = list(Path(d).glob("lib64/libcudart*"))
            print(f"  {d}: exists, libcudart: {len(contents)} files")
        else:
            print(f"  {d}: not found")
    print()

    print("--- nvidia-smi ---")
    result = _run("nvidia-smi 2>&1")
    if "not found" in result or "command not found" in result.lower():
        print(f"  {result}")
    else:
        for line in result.splitlines()[:6]:
            print(f"  {line}")
    print()

    print("--- ONNX Runtime ---")
    try:
        import onnxruntime as ort
        print(f"  version: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"  providers: {providers}")
        gpu = {"CUDAExecutionProvider", "DmlExecutionProvider", "TensorrtExecutionProvider"}
        active = gpu & set(providers)
        if active:
            print(f"  GPU active: {', '.join(active)}")
        else:
            print("  No GPU provider active.")
            print("  Check: pip install onnxruntime-gpu (not onnxruntime)")
            print("  Check: libcudnn.so.8 present and in ldconfig cache")
    except ImportError as e:
        print(f"  Import failed: {e}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    print("--- PyTorch ---")
    try:
        import torch
        print(f"  version: {torch.__version__}")
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            print(f"  CUDA: {torch.cuda.get_device_name(0)} (CC {major}.{minor})")
            print(f"  Reranker: {'cuda' if major >= 7 else 'cpu (CC too low)'}")
        else:
            print("  CUDA: not available")
    except ImportError as e:
        print(f"  Import failed: {e}")
    print()


if __name__ == "__main__":
    check()
