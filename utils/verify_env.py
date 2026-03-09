"""Environment verifier for base installs and full GPU reproduction."""

from __future__ import annotations

import argparse
import importlib
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE_IMPORTS = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "opencv-python-headless": "cv2",
    "pandas": "pandas",
    "pydicom": "pydicom",
    "plyfile": "plyfile",
    "pyhocon": "pyhocon",
    "PyMCubes": "mcubes",
    "pyvista": "pyvista",
    "PyYAML": "yaml",
    "scikit-image": "skimage",
    "scipy": "scipy",
    "SimpleITK": "SimpleITK",
    "tqdm": "tqdm",
    "wandb": "wandb",
    "kimimaro": "kimimaro",
    "crackle-codec": "crackle",
}

FULL_IMPORTS = {
    "PyTorch": "torch",
    "TIGRE": "tigre",
    "tiny-cuda-nn": "tinycudann",
    "simple-knn": "simple_knn._C",
    "diff-Xray-gaussian-rasterization-voxelization": "diff_Xray_gaussian_rasterization_voxelization",
}


def _print_status(ok: bool, label: str, detail: str = "") -> None:
    prefix = "OK" if ok else "FAIL"
    print(f"[{prefix}] {label}: {detail}" if detail else f"[{prefix}] {label}")


def _try_import(module_name: str):
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, exc


def _check_imports(packages: dict[str, str]) -> int:
    failures = 0
    for label, module_name in packages.items():
        module, error = _try_import(module_name)
        if error is not None:
            _print_status(False, label, str(error))
            failures += 1
            continue
        version = getattr(module, "__version__", None)
        detail = f"version={version}" if version else module_name
        _print_status(True, label, detail)
    return failures


def _parse_nvcc_release(version_text: str) -> str | None:
    match = re.search(r"release\s+(\d+\.\d+)", version_text)
    return match.group(1) if match else None


def _verify_supported_platform() -> int:
    system = platform.system()
    machine = platform.machine()
    supported = system == "Linux" and machine == "x86_64"
    _print_status(supported, "platform", f"{system} {machine}")
    return 0 if supported else 1


def _print_gpu_runtime_details(torch_module) -> None:
    if not torch_module.cuda.is_available():
        return
    try:
        device_count = torch_module.cuda.device_count()
        first_device = torch_module.cuda.get_device_name(0) if device_count else "none"
        _print_status(True, "CUDA devices", f"count={device_count} first={first_device}")
    except Exception as exc:
        _print_status(False, "CUDA devices", str(exc))


def _verify_repo_contract() -> int:
    failures = 0
    repo_root = Path(__file__).resolve().parents[1]

    if (repo_root / "project.yaml").exists():
        _print_status(True, "project.yaml", str(repo_root / "project.yaml"))
    else:
        _print_status(False, "project.yaml", "missing from repo root")
        failures += 1

    for module_name in ("arguments", "scene.validation", "utils.config_utils"):
        _, error = _try_import(module_name)
        if error is not None:
            _print_status(False, module_name, str(error))
            failures += 1
        else:
            _print_status(True, module_name)

    return failures


def _verify_full_gpu_stack() -> int:
    failures = _verify_supported_platform()
    failures += _check_imports(FULL_IMPORTS)

    torch, torch_error = _try_import("torch")
    if torch_error is not None:
        return failures + 1

    torch_cuda = getattr(torch.version, "cuda", None)
    if torch_cuda is None:
        _print_status(False, "torch CUDA build", "installed torch does not include CUDA support")
        failures += 1
    else:
        _print_status(True, "torch CUDA build", str(torch_cuda))

    cuda_available = bool(torch.cuda.is_available())
    _print_status(cuda_available, "torch.cuda.is_available()", str(cuda_available))
    if not cuda_available:
        failures += 1
    else:
        _print_gpu_runtime_details(torch)

    nvcc_path = shutil.which("nvcc")
    if not nvcc_path:
        _print_status(False, "nvcc", "not found on PATH")
        return failures + 1

    _print_status(True, "nvcc", nvcc_path)
    result = subprocess.run(["nvcc", "--version"], check=False, capture_output=True, text=True)
    nvcc_release = _parse_nvcc_release((result.stdout or "") + (result.stderr or ""))
    if nvcc_release is None:
        _print_status(False, "nvcc version", "unable to parse `nvcc --version` output")
        failures += 1
    else:
        _print_status(True, "nvcc version", nvcc_release)

    if torch_cuda is not None and nvcc_release is not None:
        if str(torch_cuda) != nvcc_release:
            _print_status(False, "CUDA match", f"torch.version.cuda={torch_cuda} but nvcc={nvcc_release}")
            failures += 1
        else:
            _print_status(True, "CUDA match", f"torch.version.cuda={torch_cuda}")

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also verify the GPU-only runtime required for training and evaluation.")
    args = parser.parse_args(argv)

    print(f"Python: {sys.version.split()[0]}")
    failures = 0
    failures += _check_imports(BASE_IMPORTS)
    failures += _verify_repo_contract()

    if args.full:
        failures += _verify_full_gpu_stack()

    if failures:
        print(f"\nEnvironment verification failed with {failures} issue(s).")
        return 1

    print("\nEnvironment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
