"""Environment, binary, GPU hardware, and Python dependency diagnostics for `ai-editor doctor`."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from typing import Any

from .hardware import get_hardware_profile


REQUIRED_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("certifi", "certifi"),
    ("requests", "requests"),
    ("httpx", "httpx"),
    ("pydantic", "pydantic"),
    ("PyYAML", "yaml"),
    ("pillow", "PIL"),
    ("pillow-heif", "pillow_heif"),
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("opencv-python", "cv2"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("transformers", "transformers"),
    ("huggingface-hub", "huggingface_hub"),
    ("tokenizers", "tokenizers"),
    ("safetensors", "safetensors"),
    ("sentence-transformers", "sentence_transformers"),
    ("faster-whisper", "faster_whisper"),
    ("ctranslate2", "ctranslate2"),
    ("onnxruntime", "onnxruntime"),
    ("av", "av"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("python-multipart", "python_multipart"),
    ("typer", "typer"),
    ("rich", "rich"),
)


def _probe_binary(name: str) -> dict[str, Any]:
    """Check whether a required CLI binary is on PATH and read its `-version` header."""
    path = shutil.which(name)
    if not path:
        return {"name": name, "ok": False, "path": None, "version_line": ""}

    version_line = ""
    ok = True
    try:
        proc = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        lines = (proc.stdout or proc.stderr or "").splitlines()
        version_line = lines[0].strip() if lines else ""
        if proc.returncode != 0 and not version_line:
            ok = False
    except Exception as exc:
        ok = False
        version_line = str(exc)

    return {"name": name, "ok": ok, "path": path, "version_line": version_line}


def _check_distribution(dist_name: str, import_name: str) -> dict[str, Any]:
    """Verify both installed distribution metadata and Python importability."""
    version = None
    metadata_ok = False
    try:
        version = importlib.metadata.version(dist_name)
        metadata_ok = True
    except importlib.metadata.PackageNotFoundError:
        pass

    import_ok = False
    error = ""
    try:
        if dist_name == "python-multipart":
            try:
                importlib.import_module("python_multipart")
            except ModuleNotFoundError:
                importlib.import_module("multipart")
        else:
            importlib.import_module(import_name)
        import_ok = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return {
        "distribution": dist_name,
        "import_name": import_name,
        "version": version,
        "metadata_ok": metadata_ok,
        "import_ok": import_ok,
        "ok": metadata_ok and import_ok,
        "error": error,
    }


def collect_environment_diagnostics() -> dict[str, Any]:
    """Inspect Python compatibility, FFmpeg/FFprobe binaries, GPU profile, HEIF support, and dependencies."""
    ver_info = sys.version_info
    intel_mac = sys.platform == "darwin" and platform.machine() == "x86_64"
    python_note = ""
    if (ver_info.major, ver_info.minor) < (3, 10):
        python_compatible = False
        python_note = "Python >=3.10 is required."
    elif intel_mac and (ver_info.major, ver_info.minor) >= (3, 13):
        python_compatible = False
        python_note = (
            "Intel macOS (`darwin` x86_64) requires Python 3.10–3.12 because PyTorch 2.2.2 wheels "
            "with AMD MPS support target Python <=3.12."
        )
    else:
        python_compatible = True

    ffmpeg_info = _probe_binary("ffmpeg")
    ffprobe_info = _probe_binary("ffprobe")

    hw = get_hardware_profile()
    hardware_info = {
        "platform": hw.platform,
        "arch": hw.arch,
        "cpu_count": hw.cpu_count,
        "accel_policy": hw.accel_policy,
        "torch_device": hw.torch_device,
        "has_mps": hw.has_mps,
        "has_amd_opencl": hw.has_amd_opencl,
        "opencl_device_spec": hw.opencl_device_spec,
        "discrete_gpu_name": hw.discrete_gpu_name,
        "has_videotoolbox_hwaccel": hw.has_videotoolbox_hwaccel,
        "has_scale_vt": hw.has_scale_vt,
        "has_zscale": hw.has_zscale,
        "has_filter_units_bsf": hw.has_filter_units_bsf,
        "best_hdr_encoder": hw.best_hdr_encoder,
        "best_sdr_encoder": hw.best_sdr_encoder,
    }

    heif_ok = False
    heif_error = ""
    try:
        pillow_heif = importlib.import_module("pillow_heif")
        pillow_heif.register_heif_opener()
        heif_ok = True
    except Exception as exc:
        heif_error = f"{type(exc).__name__}: {exc}"

    packages = [_check_distribution(dist_name, mod_name) for dist_name, mod_name in REQUIRED_DISTRIBUTIONS]

    optional_antigravity_version = None
    try:
        optional_antigravity_version = importlib.metadata.version("google-antigravity")
    except importlib.metadata.PackageNotFoundError:
        pass

    healthy = bool(
        python_compatible
        and ffmpeg_info["ok"]
        and ffprobe_info["ok"]
        and all(pkg["ok"] for pkg in packages)
    )

    return {
        "healthy": healthy,
        "python": {
            "version": f"{ver_info.major}.{ver_info.minor}.{ver_info.micro}",
            "executable": sys.executable,
            "platform": sys.platform,
            "arch": platform.machine(),
            "compatible": python_compatible,
            "note": python_note,
        },
        "ffmpeg": ffmpeg_info,
        "ffprobe": ffprobe_info,
        "heif": {"ok": heif_ok, "error": heif_error},
        "hardware": hardware_info,
        "packages": packages,
        "optional": {
            "google-antigravity": optional_antigravity_version,
        },
    }
