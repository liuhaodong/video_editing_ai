"""Unified macOS AMD GPU (MPS + OpenCL), Apple VideoToolbox, and multi-core hardware policy helpers."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
from typing import Iterator, Optional


_LOCK = threading.RLock()
_VT_HDR_ENCODE_LOCK = threading.Lock()

_VALID_ACCEL_POLICIES = frozenset({"auto", "mps", "opencl", "cpu"})

_PROFILE_CACHE: dict[tuple[str, str], "HardwareProfile"] = {}
_FILTERS_CACHE: Optional[frozenset[str]] = None
_BSFS_CACHE: Optional[frozenset[str]] = None
_HWACCELS_CACHE: Optional[frozenset[str]] = None
_ENCODER_CACHE: dict[tuple[bool, str], str] = {}
_VT_QSCALE_CACHE: dict[tuple[str, str], bool] = {}
_MACOS_GPUS_CACHE: Optional[list[dict[str, str]]] = None
_OPENCL_STATE: Optional[tuple[bool, str, bool]] = None


@dataclass(frozen=True)
class HardwareProfile:
    """Snapshot of detected compute GPUs, FFmpeg hardware codecs/filters, and runtime policy."""

    platform: str
    arch: str
    cpu_count: int
    accel_policy: str
    has_mps: bool
    torch_device: str
    has_amd_opencl: bool
    opencl_device_spec: str
    has_videotoolbox_hwaccel: bool
    has_scale_vt: bool
    has_zscale: bool
    has_filter_units_bsf: bool
    best_hdr_encoder: str
    best_sdr_encoder: str
    vt_supports_qscale_hdr: bool
    vt_supports_qscale_sdr: bool
    discrete_gpu_name: str


def reset_hardware_cache() -> None:
    """Clear cached hardware probes (primarily used by tests when changing AI_EDITOR_ACCEL)."""
    global _FILTERS_CACHE, _BSFS_CACHE, _HWACCELS_CACHE, _MACOS_GPUS_CACHE, _OPENCL_STATE
    with _LOCK:
        _PROFILE_CACHE.clear()
        _FILTERS_CACHE = None
        _BSFS_CACHE = None
        _HWACCELS_CACHE = None
        _ENCODER_CACHE.clear()
        _VT_QSCALE_CACHE.clear()
        _MACOS_GPUS_CACHE = None
        _OPENCL_STATE = None


def get_accel_policy() -> str:
    """Return normalized acceleration policy from AI_EDITOR_ACCEL ('auto'|'mps'|'opencl'|'cpu')."""
    raw = os.environ.get("AI_EDITOR_ACCEL", "auto").strip().lower()
    return raw if raw in _VALID_ACCEL_POLICIES else "auto"


def get_io_worker_count(default_cap: int = 8) -> int:
    """Return bounded worker count respecting AI_EDITOR_MAX_WORKERS override."""
    raw = os.environ.get("AI_EDITOR_MAX_WORKERS", "").strip()
    if raw:
        try:
            parsed = int(raw)
            if parsed >= 1:
                return parsed
        except ValueError:
            pass
    cpus = max(1, os.cpu_count() or 8)
    return max(1, min(default_cap, cpus))


def select_torch_device() -> str:
    """Select 'mps' when AI_EDITOR_ACCEL allows it and PyTorch MPS backend is operational, else 'cpu'."""
    policy = get_accel_policy()
    if policy not in ("auto", "mps"):
        return "cpu"
    try:
        import torch

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        ):
            return "mps"
    except Exception:
        pass
    return "cpu"


def _detect_macos_gpus() -> list[dict[str, str]]:
    """Query macOS display adapters via system_profiler once and cache the result."""
    global _MACOS_GPUS_CACHE
    with _LOCK:
        if _MACOS_GPUS_CACHE is not None:
            return _MACOS_GPUS_CACHE
        if sys.platform != "darwin":
            _MACOS_GPUS_CACHE = []
            return _MACOS_GPUS_CACHE
        gpus: list[dict[str, str]] = []
        try:
            proc = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                check=False,
                timeout=6,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                payload = json.loads(proc.stdout)
                for item in payload.get("SPDisplaysDataType") or []:
                    if not isinstance(item, dict):
                        continue
                    gpus.append(
                        {
                            "name": str(item.get("sppci_model") or item.get("_name") or ""),
                            "vendor": str(item.get("spdisplays_vendor") or ""),
                            "vram": str(item.get("spdisplays_vram") or item.get("spdisplays_vram_shared") or ""),
                        }
                    )
        except Exception:
            gpus = []
        _MACOS_GPUS_CACHE = gpus
        return _MACOS_GPUS_CACHE


def _find_amd_gpu_index_and_name(gpus: list[dict[str, str]]) -> tuple[int, str]:
    """Return (index, name) of the discrete AMD GPU on dual-GPU Intel+AMD Macs, else (-1, '')."""
    for idx, g in enumerate(gpus):
        name = g.get("name", "")
        vendor = g.get("vendor", "")
        combined = f"{name} {vendor}".lower()
        if "amd" in combined or "radeon" in combined or "0x1002" in combined:
            return idx, name or "AMD Radeon"
    return -1, ""


def configure_opencv_amd_opencl() -> bool:
    """
    Configure OpenCV OpenCL runtime, binding macOS discrete AMD GPU via 'Apple:GPU:<idx>' when present.
    Never sets 'AMD:GPU:' because Apple's OpenCL platform name on macOS is 'Apple'.
    """
    global _OPENCL_STATE
    policy = get_accel_policy()
    with _LOCK:
        try:
            import cv2
        except Exception:
            _OPENCL_STATE = (False, "", False)
            return False

        if policy in ("cpu", "mps"):
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass
            _OPENCL_STATE = (False, "", False)
            return False

        if not cv2.ocl.haveOpenCL():
            _OPENCL_STATE = (False, "", False)
            return False

        amd_idx = -1
        if sys.platform == "darwin" and not os.environ.get("OPENCV_OPENCL_DEVICE"):
            gpus = _detect_macos_gpus()
            amd_idx, _ = _find_amd_gpu_index_and_name(gpus)
            if len(gpus) >= 2 and amd_idx >= 1:
                # Apple's OpenCL platform name is "Apple", not "AMD".
                os.environ["OPENCV_OPENCL_DEVICE"] = f"Apple:GPU:{amd_idx}"

        spec = os.environ.get("OPENCV_OPENCL_DEVICE", "")
        try:
            cv2.ocl.setUseOpenCL(True)
            active = bool(cv2.ocl.useOpenCL())
        except Exception:
            active = False

        if not active:
            _OPENCL_STATE = (False, spec, False)
            return False

        has_amd = False
        try:
            dev = cv2.ocl.Device.getDefault()
            dev_desc = f"{dev.name()} {dev.vendorName()}".lower()
            has_amd = ("amd" in dev_desc) or ("radeon" in dev_desc) or (amd_idx >= 1 and spec.startswith("Apple:GPU:"))
        except Exception:
            has_amd = amd_idx >= 1 and spec.startswith("Apple:GPU:")

        _OPENCL_STATE = (True, spec, has_amd)
        return True


def _load_ffmpeg_filters() -> frozenset[str]:
    global _FILTERS_CACHE
    with _LOCK:
        if _FILTERS_CACHE is not None:
            return _FILTERS_CACHE
        found: set[str] = set()
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-filters"],
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
            for line in (proc.stdout or "").splitlines():
                tokens = line.strip().split()
                if len(tokens) >= 2 and not tokens[0].startswith("-"):
                    # Typical line: "TS. scale_vt          V->V       Scale VideoToolbox frames"
                    if len(tokens[0]) <= 4 and all(ch in "TSC." for ch in tokens[0]):
                        found.add(tokens[1])
                    else:
                        found.add(tokens[0])
        except Exception:
            pass
        _FILTERS_CACHE = frozenset(found)
        return _FILTERS_CACHE


def _load_ffmpeg_bsfs() -> frozenset[str]:
    global _BSFS_CACHE
    with _LOCK:
        if _BSFS_CACHE is not None:
            return _BSFS_CACHE
        found: set[str] = set()
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-bsfs"],
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
            for line in (proc.stdout or "").splitlines():
                tok = line.strip()
                if tok and not tok.endswith(":") and " " not in tok:
                    found.add(tok)
        except Exception:
            pass
        _BSFS_CACHE = frozenset(found)
        return _BSFS_CACHE


def _load_ffmpeg_hwaccels() -> frozenset[str]:
    global _HWACCELS_CACHE
    with _LOCK:
        if _HWACCELS_CACHE is not None:
            return _HWACCELS_CACHE
        found: set[str] = set()
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
            for line in (proc.stdout or "").splitlines():
                tok = line.strip()
                if tok and not tok.endswith(":") and " " not in tok:
                    found.add(tok)
        except Exception:
            pass
        _HWACCELS_CACHE = frozenset(found)
        return _HWACCELS_CACHE


def has_ffmpeg_filter(filter_name: str) -> bool:
    """Return True if the named filter is compiled into the local ffmpeg binary (cached)."""
    return filter_name in _load_ffmpeg_filters()


def has_ffmpeg_bsf(bsf_name: str) -> bool:
    """Return True if the named bitstream filter is compiled into local ffmpeg (cached)."""
    return bsf_name in _load_ffmpeg_bsfs()


def has_ffmpeg_hwaccel(name: str) -> bool:
    """Return True if hardware accel is supported by local ffmpeg and permitted by AI_EDITOR_ACCEL."""
    if get_accel_policy() == "cpu":
        return False
    return name in _load_ffmpeg_hwaccels()


def probe_hw_encoder(encoder: str, pix_fmt: str = "yuv420p", extra_args: Optional[list[str]] = None) -> bool:
    """Test whether a specific hardware encoder accepts a tiny synthetic encode."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "nullsrc=s=256x256:d=0.1",
        "-c:v",
        encoder,
        "-pix_fmt",
        pix_fmt,
        *(extra_args or []),
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=10)
        return proc.returncode == 0
    except Exception:
        return False


def supports_videotoolbox_qscale(encoder: str, pix_fmt: str) -> bool:
    """Cached check whether VideoToolbox accepts -q:v alongside -b:v."""
    if get_accel_policy() == "cpu":
        return False
    key = (encoder, pix_fmt)
    with _LOCK:
        if key not in _VT_QSCALE_CACHE:
            extra = ["-profile:v", "main10"] if encoder == "hevc_videotoolbox" else []
            _VT_QSCALE_CACHE[key] = probe_hw_encoder(
                encoder,
                pix_fmt=pix_fmt,
                extra_args=[*extra, "-q:v", "68", "-b:v", "28M", "-allow_sw", "1"],
            )
        return _VT_QSCALE_CACHE[key]


def detect_best_video_encoder(hdr: bool = True) -> str:
    """Cached hardware/software encoder selection honoring AI_EDITOR_ACCEL."""
    policy = get_accel_policy()
    if policy == "cpu":
        return "libx265" if hdr else "libx264"

    key = (hdr, policy)
    with _LOCK:
        if key in _ENCODER_CACHE:
            return _ENCODER_CACHE[key]

        if hdr:
            candidates = [
                ("hevc_videotoolbox", "p010le", ["-profile:v", "main10", "-allow_sw", "1"]),
                ("hevc_nvenc", "p010le", []),
                ("hevc_amf", "p010le", []),
                ("hevc_qsv", "p010le", []),
            ]
            fallback = "libx265"
        else:
            candidates = [
                ("h264_videotoolbox", "yuv420p", ["-allow_sw", "1"]),
                ("h264_nvenc", "yuv420p", []),
                ("h264_amf", "yuv420p", []),
                ("h264_qsv", "yuv420p", []),
            ]
            fallback = "libx264"

        for enc_name, p_fmt, ex in candidates:
            if probe_hw_encoder(enc_name, pix_fmt=p_fmt, extra_args=ex):
                _ENCODER_CACHE[key] = enc_name
                return enc_name

        _ENCODER_CACHE[key] = fallback
        return fallback


def get_hardware_profile() -> HardwareProfile:
    """Return cached HardwareProfile for the current environment & AI_EDITOR_ACCEL setting."""
    policy = get_accel_policy()
    ocl_env = os.environ.get("OPENCV_OPENCL_DEVICE", "")
    cache_key = (policy, ocl_env)
    with _LOCK:
        cached = _PROFILE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        torch_dev = select_torch_device()
        has_mps = torch_dev == "mps"

        configure_opencv_amd_opencl()
        ocl_active, ocl_spec, has_amd_ocl = _OPENCL_STATE or (False, "", False)

        vt_hwaccel = has_ffmpeg_hwaccel("videotoolbox")
        has_svt = vt_hwaccel and has_ffmpeg_filter("scale_vt")
        has_zs = has_ffmpeg_filter("zscale")
        has_fu = has_ffmpeg_bsf("filter_units")

        best_hdr = detect_best_video_encoder(hdr=True)
        best_sdr = detect_best_video_encoder(hdr=False)
        vt_q_hdr = supports_videotoolbox_qscale("hevc_videotoolbox", "p010le") if best_hdr == "hevc_videotoolbox" else False
        vt_q_sdr = supports_videotoolbox_qscale("h264_videotoolbox", "yuv420p") if best_sdr == "h264_videotoolbox" else False

        _, discrete_name = _find_amd_gpu_index_and_name(_detect_macos_gpus())

        prof = HardwareProfile(
            platform=sys.platform,
            arch=platform.machine(),
            cpu_count=max(1, os.cpu_count() or 1),
            accel_policy=policy,
            has_mps=has_mps,
            torch_device=torch_dev,
            has_amd_opencl=bool(ocl_active and has_amd_ocl),
            opencl_device_spec=ocl_spec,
            has_videotoolbox_hwaccel=vt_hwaccel,
            has_scale_vt=has_svt,
            has_zscale=has_zs,
            has_filter_units_bsf=has_fu,
            best_hdr_encoder=best_hdr,
            best_sdr_encoder=best_sdr,
            vt_supports_qscale_hdr=vt_q_hdr,
            vt_supports_qscale_sdr=vt_q_sdr,
            discrete_gpu_name=discrete_name,
        )
        _PROFILE_CACHE[cache_key] = prof
        return prof


@contextlib.contextmanager
def videotoolbox_hdr_encode_guard(hdr: bool, v_enc_args: list[str]) -> Iterator[None]:
    """
    Serialize macOS Apple T2 `hevc_videotoolbox` HDR Main10 encodes cleanly.

    Concurrent `hevc_videotoolbox -profile:v main10` FFmpeg processes on dual-GPU Intel+T2
    Macs spill secondary sessions onto a second encoder path with differing `hvcC` flags
    (`0102 60...` vs `0102 20...`), breaking Step 3 `-f concat -c copy`. Serializing only
    the HDR VT encode lets CPU prep overlap across workers while keeping 100% uniform `hvcC`.
    """
    if sys.platform == "darwin" and hdr and "hevc_videotoolbox" in v_enc_args:
        with _VT_HDR_ENCODE_LOCK:
            yield
    else:
        yield


@contextlib.contextmanager
def caffeinate_guard() -> Iterator[None]:
    """Prevent macOS idle system sleep during long multi-minute GPU/VideoToolbox batch renders."""
    if sys.platform != "darwin" or shutil.which("caffeinate") is None:
        yield
        return

    proc: Optional[subprocess.Popen[bytes]] = None
    try:
        proc = subprocess.Popen(
            ["caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        proc = None

    try:
        yield
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
