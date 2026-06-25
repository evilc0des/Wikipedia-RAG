import sys


def check():
    print("=== GPU Provider Check ===")

    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")

        gpu = {"CUDAExecutionProvider", "DmlExecutionProvider", "TensorrtExecutionProvider"}
        active = gpu & set(providers)
        if active:
            print(f"GPU: {', '.join(active)}")
        else:
            on_windows = sys.platform == "win32"
            print("No GPU provider active. Install:")
            if on_windows:
                print("  pip install onnxruntime-directml  # DirectML (Windows)")
            print("  pip install onnxruntime-gpu        # CUDA (requires NVIDIA GPU + CUDA runtime)")
    except ImportError:
        print("onnxruntime not installed. Run: pip install onnxruntime-gpu")

    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            print(f"PyTorch CUDA: {torch.cuda.get_device_name(0)} (CC {major}.{minor})")
            print(f"  Reranker: {'cuda' if major >= 7 else 'cpu (CC too low)'}")
        else:
            print("PyTorch CUDA: not available (reranker -> CPU)")
    except ImportError:
        print("PyTorch not installed")


if __name__ == "__main__":
    check()
