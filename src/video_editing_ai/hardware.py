"""Unified GPU (NVIDIA CUDA/NVENC/NVDEC, macOS AMD MPS + OpenCL, VideoToolbox) and multi-core hardware policy helpers."""

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
_NVENC_ENCODE_SEM = threading.BoundedSemaphore(4)
_CUDA_NVDEC_SEM = threading.BoundedSemaphore(4)

_VALID_ACCEL_POLICIES = frozenset({"auto", "cuda", "mps", "opencl", "cpu"})

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
    has_cuda: bool = False
    has_cuda_hwaccel: bool = False
    cuda_vram_gb: float = 0.0


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
    """Return normalized acceleration policy from AI_EDITOR_ACCEL ('auto'|'cuda'|'mps'|'opencl'|'cpu')."""
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


def get_render_worker_count(hw: Optional[HardwareProfile] = None) -> int:
    """Return optimal concurrent V1/V2 render worker count for the active GPU/CPU profile."""
    raw = os.environ.get("AI_EDITOR_MAX_WORKERS", "").strip()
    if raw:
        try:
            parsed = int(raw)
            if parsed >= 1:
                return parsed
        except ValueError:
            pass
    prof = hw or get_hardware_profile()
    if prof.platform == "darwin":
        return min(5, max(1, prof.cpu_count // 4))
    if prof.has_cuda or "nvenc" in prof.best_hdr_encoder:
        if prof.cuda_vram_gb >= 12.0:
            return min(8, max(6, prof.cpu_count // 4))
        return min(6, max(4, prof.cpu_count // 5))
    return min(8, max(2, prof.cpu_count // 4))


def select_torch_device(policy: Optional[str] = None) -> str:
    """Select 'cuda', 'mps', or 'cpu' honoring AI_EDITOR_ACCEL policy and operational backends."""
    eff = policy.strip().lower() if policy else get_accel_policy()
    if eff not in _VALID_ACCEL_POLICIES:
        eff = "auto"
    if eff == "cpu":
        return "cpu"
    try:
        import torch

        if eff in ("auto", "cuda"):
            if torch.cuda.is_available():
                return "cuda"
            if eff == "cuda":
                return "cpu"
        if eff in ("auto", "mps", "opencl"):
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
                and torch.backends.mps.is_available()
            ):
                return "mps"
    except Exception:
        pass
    return "cpu"


def _detect_cuda_device_info(has_cuda: bool) -> tuple[str, float]:
    """Return (device_name, total_vram_gb) for active CUDA GPU or nvidia-smi adapter."""
    if has_cuda:
        try:
            import torch

            name = str(torch.cuda.get_device_name(0) or "").strip()
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(float(props.total_memory) / float(1024 ** 3), 2)
            return name, vram_gb
        except Exception:
            pass
    if sys.platform != "darwin" and shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
                timeout=4,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                first_line = proc.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first_line.split(",")]
                name = parts[0] if parts else ""
                vram_gb = round(float(parts[1]) / 1024.0, 2) if len(parts) > 1 and parts[1] else 0.0
                return name, vram_gb
        except Exception:
            pass
    return "", 0.0


def get_cuda_free_vram_gb(device_idx: int = 0) -> float:
    """Return live free VRAM in GiB for CUDA device `device_idx` (0.0 if unavailable)."""
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device_idx)
            return round(float(free_bytes) / float(1024 ** 3), 2)
    except Exception:
        pass
    if sys.platform != "darwin" and shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
                timeout=4,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                first_val = proc.stdout.strip().splitlines()[0].strip()
                if first_val:
                    return round(float(first_val) / 1024.0, 2)
        except Exception:
            pass
    return 0.0


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

        torch_dev = select_torch_device(policy=policy)
        has_cuda = torch_dev == "cuda"
        has_mps = torch_dev == "mps"
        cuda_hwaccel = (policy != "cpu") and has_ffmpeg_hwaccel("cuda")

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
        cuda_vram_gb = 0.0
        if policy != "cpu":
            cuda_name, cuda_vram_gb = _detect_cuda_device_info(has_cuda)
            if not discrete_name and cuda_name:
                discrete_name = cuda_name

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
            has_cuda=has_cuda,
            has_cuda_hwaccel=cuda_hwaccel,
            cuda_vram_gb=cuda_vram_gb if has_cuda else 0.0,
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
def nvenc_encode_guard(v_enc_args: list[str] | tuple[str, ...]) -> Iterator[None]:
    """
    Bound GPU-wide concurrent NVENC sessions across photo, video, map, and concat workers.

    Prevents simultaneous PyTorch CUDA workloads + FFmpeg NVENC/WDDM buffer allocations
    from exceeding VRAM or hardware session limits and causing silent frame/slice tears.
    """
    if any("nvenc" in str(arg).lower() for arg in v_enc_args):
        with _NVENC_ENCODE_SEM:
            yield
    else:
        yield


@contextlib.contextmanager
def cuda_nvdec_guard(hw_prefix: list[str] | tuple[str, ...]) -> Iterator[None]:
    """Bound concurrent `-hwaccel cuda` NVDEC decode workers so 12-16 thread analyzers stay within VRAM/NVDEC limits."""
    if any(str(tok).lower() == "cuda" for tok in hw_prefix):
        with _CUDA_NVDEC_SEM:
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
