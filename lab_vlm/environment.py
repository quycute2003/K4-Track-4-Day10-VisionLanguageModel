"""Check optional packages without importing Torch or loading a GPU model."""
import importlib.metadata
import subprocess
import sys

from packaging.version import Version


def remove_incompatible_colab_torchao():
    """The lab uses BitsAndBytes; remove only TorchAO versions unsupported by PEFT."""
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return False
    if Version(version) >= Version("0.16.0"):
        return False
    print(f"Colab có torchao {version}, không tương thích PEFT 0.21.2. "
          "Gỡ backend tùy chọn này; bài lab dùng BitsAndBytes.", flush=True)
    result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                            check=True, capture_output=True, text=True)
    print(result.stdout, flush=True)
    importlib.invalidate_caches()
    return True


def check_vit_lora_backend(cfg):
    """Fail before allocating a model when PEFT cannot attach LoRA to ordinary ViT Linear layers."""
    if cfg.get("lora", {}).get("targets") != "llm+vit":
        return
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return
    if Version(version) < Version("0.16.0"):
        raise RuntimeError(f"torchao {version} không tương thích PEFT: "
                           "gỡ torchao bằng pip uninstall -y torchao rồi chạy lại riêng E3. "
                           "Bài lab dùng BitsAndBytes; checkpoint E1/E2 vẫn dùng được.")
