"""
Interactive Python Environment Setup Script
Optimized for modern ML workflows
Includes automatic GPU detection and TORCH LOCKING to prevent downgrades
Supports uv (fast) with automatic fallback to pip
"""

import subprocess
import sys
import argparse
from pathlib import Path

VENV_DIR = ".venv"
TORCH_LOCK_FILE = Path(VENV_DIR) / "torch.lock"
USE_VENV = True
USE_UV = False  # Set automatically by detect_uv()
GPU_AVAILABLE = False
CUDA_VERSION = "cu121"
UPGRADE = False
REINSTALL_TORCH = False

BASE_PACKAGES = [
    "matplotlib",
    "seaborn",
    "IPython",
    "IProgress",
    "ipykernel",
    "pandas",
    "tqdm",
    "numpy",
    "scikit-learn",
    "plotly",
    "jupyter",
    "ipywidgets",
    "pyarrow",
    "fastparquet", 
]
    
CUSTOM_PACKAGES = [
    "pysbd",
    "nltk"
]

# Packages for the classification server
CLASSIFICATION_PACKAGES = [
    "tensorboardX",
    "transformers",
    "evaluate",
    "datasets",
    "seqeval",
    "accelerate",
]

# For the old "install all" option, kept for compatibility if needed
# but the new menu provides more granular control.
PACKAGES = CLASSIFICATION_PACKAGES + BASE_PACKAGES + CUSTOM_PACKAGES


# ---------------------------------------------------------------------------
# uv detection
# ---------------------------------------------------------------------------


def detect_uv() -> bool:
    """Return True if uv is available on PATH."""
    global USE_UV
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"⚡ uv detected ({version}) — using uv for package management.")
            USE_UV = True
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("   uv not found — falling back to pip.")
    USE_UV = False
    return False


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def detect_nvidia_gpu():
    """Detect if NVIDIA GPU is available and extract CUDA version dynamically."""
    global GPU_AVAILABLE, CUDA_VERSION

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            GPU_AVAILABLE = True
            print("✅ NVIDIA GPU detected!")

            try:
                gpu_info = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if gpu_info.returncode == 0:
                    print(f"   GPU: {gpu_info.stdout.strip()}")
            except Exception:
                pass

            try:
                cuda_info = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                import re

                match = re.search(r"CUDA Version: (\d+)\.(\d+)", cuda_info.stdout)
                if match:
                    major, minor = match.groups()
                    CUDA_VERSION = f"cu{major}{minor}"
                    print(f"   Detected CUDA version: {major}.{minor}")
                else:
                    print(
                        f"   Could not parse CUDA version, using default: {CUDA_VERSION}"
                    )
                print(f"   Using PyTorch wheel: {CUDA_VERSION}")
            except Exception as e:
                print(
                    f"   Could not detect CUDA version: {e}, using default: {CUDA_VERSION}"
                )

            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    GPU_AVAILABLE = False
    return False


def detect_amd_gpu():
    """Detect if AMD GPU is available with ROCm."""
    try:
        result = subprocess.run(
            ["rocm-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print("✅ AMD GPU with ROCm detected!")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def get_supported_cuda_version(detected: str) -> str:
    """
    Clamp the detected CUDA version to the latest wheel PyTorch actually
    publishes. Newer drivers are backward-compatible, so the highest
    supported wheel always works.

    Update SUPPORTED_CUDA_VERSIONS when PyTorch adds new wheels.
    See: https://download.pytorch.org/whl/torch/
    """
    SUPPORTED_CUDA_VERSIONS = ["cu118", "cu121", "cu124", "cu126", "cu128"]

    if detected in SUPPORTED_CUDA_VERSIONS:
        return detected

    def _ver_num(tag: str) -> int:
        try:
            return int(tag.replace("cu", ""))
        except ValueError:
            return 0

    detected_num = _ver_num(detected)
    supported_nums = [_ver_num(v) for v in SUPPORTED_CUDA_VERSIONS]

    if detected_num > max(supported_nums):
        clamped = SUPPORTED_CUDA_VERSIONS[-1]
        print(
            f"   ⚠️  CUDA {detected} has no PyTorch wheel yet. "
            f"Falling back to {clamped} (fully compatible with your driver)."
        )
        return clamped

    for ver, num in zip(reversed(SUPPORTED_CUDA_VERSIONS), reversed(supported_nums)):
        if detected_num >= num:
            print(f"   ⚠️  No exact wheel for {detected}, using {ver}.")
            return ver

    return SUPPORTED_CUDA_VERSIONS[-1]


def get_pytorch_install_args() -> list[str]:
    """Return the PyTorch package list + index-url args for the current hardware."""
    if GPU_AVAILABLE == "nvidia":
        wheel_tag = get_supported_cuda_version(CUDA_VERSION)
        return [
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            f"https://download.pytorch.org/whl/{wheel_tag}",
        ]
    elif GPU_AVAILABLE == "amd":
        return [
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/rocm6.2",
        ]
    else:
        return [
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ]


# ---------------------------------------------------------------------------
# Installer helpers
# ---------------------------------------------------------------------------


def _build_install_cmd(
    packages: list[str], extra_args: list[str] | None = None
) -> list[str]:
    """
    Build the full install command as a list (no shell=True needed).

    uv pip install  → uv pip install [--upgrade] <pkgs> [extra_args]
    pip install     → <venv>/bin/pip install [--upgrade] <pkgs> [extra_args]
    """
    extra_args = extra_args or []

    if USE_UV:
        cmd = ["uv", "pip", "install"]
        if USE_VENV:
            # Tell uv which venv to target explicitly
            cmd += ["--python", _python_executable()]
        if UPGRADE:
            cmd.append("--upgrade")
        cmd += packages + extra_args
    else:
        cmd = [_pip_executable()]
        cmd += ["install"]
        if UPGRADE:
            cmd.append("--upgrade")
        cmd += packages + extra_args

    return cmd


def _pip_executable() -> str:
    """Path to the venv pip (or bare 'pip' when not using a venv)."""
    if not USE_VENV:
        return "pip"
    if sys.platform == "win32":
        return f"{VENV_DIR}\\Scripts\\pip.exe"
    return f"{VENV_DIR}/bin/pip"


def _python_executable() -> str:
    """Path to the venv python (or the current interpreter)."""
    if not USE_VENV:
        return sys.executable
    if sys.platform == "win32":
        return f"{VENV_DIR}\\Scripts\\python.exe"
    return f"{VENV_DIR}/bin/python"


# Keep old name for any callers that still reference it
def get_pip_executable() -> str:
    return _pip_executable()


def install_packages(package_list: list[str], description: str):
    """Install a list of packages using uv or pip."""
    print(f"📦 Installing {description}...")
    cmd = _build_install_cmd(package_list)
    print(f"   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"✅ {description} installed successfully.")
    else:
        print(f"❌ Failed to install some {description}.")


def install_pytorch():
    """Install PyTorch with appropriate GPU support."""
    print("📦 Installing PyTorch...")
    torch_args = get_pytorch_install_args()

    # Split packages from index-url args so _build_install_cmd can position them correctly
    # torch_args looks like: ["torch", "torchvision", "torchaudio", "--index-url", "<url>"]
    try:
        idx = torch_args.index("--index-url")
        packages = torch_args[:idx]
        extra = torch_args[idx:]
    except ValueError:
        packages = torch_args
        extra = []

    cmd = _build_install_cmd(packages, extra_args=extra)
    print(f"   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        # Record installed version and lock it
        try:
            if USE_UV:
                version_result = subprocess.run(
                    ["uv", "pip", "show", "torch", "--python", _python_executable()],
                    capture_output=True,
                    text=True,
                )
            else:
                version_result = subprocess.run(
                    [_pip_executable(), "show", "torch"],
                    capture_output=True,
                    text=True,
                )
            if "Version:" in version_result.stdout:
                version = version_result.stdout.split("Version: ")[1].split("\n")[0]
                TORCH_LOCK_FILE.write_text(version)
                print(f"🧱 PyTorch {version} locked to {TORCH_LOCK_FILE}")
        except Exception:
            pass

        if GPU_AVAILABLE == "nvidia":
            print(f"✅ PyTorch (NVIDIA GPU {CUDA_VERSION}) installed successfully.")
        elif GPU_AVAILABLE == "amd":
            print("✅ PyTorch (AMD ROCm) installed successfully.")
        else:
            print("✅ PyTorch (CPU) installed successfully.")
    else:
        print("❌ Failed to install PyTorch.")


def is_torch_locked() -> bool:
    """Check if PyTorch is locked."""
    return TORCH_LOCK_FILE.exists()


def create_venv():
    """Create the virtual environment if it doesn't exist."""
    venv_path = Path(VENV_DIR)
    if not venv_path.exists():
        print(f"🛠️ Creating virtual environment in '{VENV_DIR}'...")
        try:
            if USE_UV:
                subprocess.run(["uv", "venv", VENV_DIR], check=True)
            else:
                subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
            print("✅ Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            sys.exit(1)
    else:
        print(f"✓ Found existing virtual environment: '{VENV_DIR}'")


# ---------------------------------------------------------------------------
# Menu / UI
# ---------------------------------------------------------------------------


def show_menu():
    """Display interactive menu."""
    print("\n" + "=" * 60)
    print("🐍 INTERACTIVE ENVIRONMENT SETUP")
    print("=" * 60)
    venv_status = (
        f"ACTIVE (in ./{VENV_DIR})" if USE_VENV else "INACTIVE (global site-packages)"
    )
    print(f"Virtual Environment : {venv_status}")
    installer = "uv ⚡" if USE_UV else "pip"
    print(f"Package Manager     : {installer}")
    platform_info = "Windows" if sys.platform == "win32" else "Linux/WSL/Mac"
    print(f"Platform            : {platform_info}")

    if GPU_AVAILABLE == "nvidia":
        gpu_status = f"GPU: Detected ({CUDA_VERSION})"
    elif GPU_AVAILABLE == "amd":
        gpu_status = "GPU: AMD ROCm detected"
    else:
        gpu_status = "GPU: Not detected (CPU-only)"
    print(f"{gpu_status}")

    torch_status = (
        "🧱 PyTorch is LOCKED" if is_torch_locked() else "PyTorch is unlocked"
    )
    print(f"Torch Status        : {torch_status}")

    print("\nOptions:")
    print("  0. Basic setup (includes custom packages)")
    print("  1. Install ML Packages (Classification Server)")
    print("  2. Install ML Packages (Full Training Setup)")
    print("  3. Check current installation")
    print("  4. Reinstall PyTorch (unlock and reinstall)")
    print("  5. Exit")
    print("-" * 60)


def check_installation():
    """Check what's currently installed."""
    print("\n🔍 Checking current installation...")
    python_exec = _python_executable()
    print(f"   Using Python: {python_exec}")

    def get_package_version(pkg_name):
        cmd = f'{python_exec} -c "import {pkg_name}; print({pkg_name}.__version__)"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()

    packages_to_check = ["torch", "pandas", "pyarrow", "transformers", "sklearn"]
    for pkg in packages_to_check:
        version = get_package_version(pkg)
        print(f"   {pkg}: {version if version else 'Not installed'}")

    print("\n🎮 Checking GPU support...")
    gpu_check_cmd = (
        f'{python_exec} -c "'
        "import torch; "
        "print(f'CUDA available: {torch.cuda.is_available()}'); "
        "print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
        '"'
    )
    subprocess.run(gpu_check_cmd, shell=True)

    print("\n📦 Checking Parquet support...")
    parquet_check_cmd = (
        f'{python_exec} -c "'
        "import pandas as pd, sys; "
        "pd.io.parquet.get_engine('auto'); "
        "print('✅ Parquet engine available')"
        '"'
    )
    subprocess.run(parquet_check_cmd, shell=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    global USE_VENV, GPU_AVAILABLE, UPGRADE, REINSTALL_TORCH

    parser = argparse.ArgumentParser(
        description="Interactive environment setup script with torch locking."
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Install packages in the global environment instead of the virtual environment.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Do not use upgrade flags when installing packages.",
    )
    parser.add_argument(
        "--reinstall-torch",
        action="store_true",
        help="Reinstall PyTorch even if locked.",
    )
    args = parser.parse_args()

    if args.no_venv:
        USE_VENV = False
    if args.upgrade:
        UPGRADE = True
    if args.reinstall_torch:
        REINSTALL_TORCH = True

    print("\n🔍 Detecting package manager...")
    detect_uv()

    print("\n🔍 Detecting hardware...")
    if detect_nvidia_gpu():
        GPU_AVAILABLE = "nvidia"
    elif detect_amd_gpu():
        GPU_AVAILABLE = "amd"
    else:
        print("   No GPU detected. Will use CPU-only PyTorch.")

    if USE_VENV:
        create_venv()

    while True:
        show_menu()
        choice = input("\nEnter your choice (0-5): ").strip()

        if choice == "0":
            print("\nBasic setup starting...")
            install_packages(BASE_PACKAGES, "base packages")
            install_packages(CUSTOM_PACKAGES, "custom packages")
            print("\n✅ Basic setup complete!")
            sys.exit(0)

        elif choice == "1":
            print("\nSetting up for Classification Server...")
            if is_torch_locked() and not REINSTALL_TORCH:
                print("🧱 PyTorch is already locked. Skipping PyTorch install.")
            else:
                install_pytorch()
            install_packages(CLASSIFICATION_PACKAGES, "classification packages")
            install_packages(CUSTOM_PACKAGES, "custom packages")
            install_packages(BASE_PACKAGES, "base packages")
            print("\n✅ Classification Server setup complete!")
            sys.exit(0)

        elif choice == "2":
            print("\nStarting Full Training Setup...")
            if is_torch_locked() and not REINSTALL_TORCH:
                print("🧱 PyTorch is already locked. Skipping PyTorch install.")
            else:
                install_pytorch()
            install_packages(CLASSIFICATION_PACKAGES, "classification packages")
            install_packages(CUSTOM_PACKAGES, "custom packages")
            install_packages(BASE_PACKAGES, "base packages")
            print("\n✅ Full Training Environment setup complete!")
            sys.exit(0)

        elif choice == "3":
            check_installation()

        elif choice == "4":
            print("\n🔄 Reinstalling PyTorch...")
            TORCH_LOCK_FILE.unlink(missing_ok=True)
            install_pytorch()

        else:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
