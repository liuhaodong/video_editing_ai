<#
.SYNOPSIS
    Turnkey Windows PowerShell setup script for video-editing-ai.
.DESCRIPTION
    Checks/installs FFmpeg, creates a Python virtual environment, installs all runtime/GPU/dev
    dependencies from pyproject.toml, and runs `ai-editor doctor` to verify the environment.
#>
[CmdletBinding()]
param(
    [switch]$SkipSystemDeps,
    [switch]$NoDev,
    [switch]$Editable,
    [switch]$WithAntigravity,
    [switch]$SyncSkill,
    [switch]$NoSyncSkill,
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

Write-Host "==> Setting up video-editing-ai in $RepoRoot"

if (-not $SkipSystemDeps) {
    $hasFfmpeg = $null -ne (Get-Command ffmpeg -ErrorAction SilentlyContinue)
    $hasFfprobe = $null -ne (Get-Command ffprobe -ErrorAction SilentlyContinue)
    if (-not $hasFfmpeg -or -not $hasFfprobe) {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            Write-Host "==> Installing FFmpeg via winget (Gyan.FFmpeg)"
            winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
        } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
            Write-Host "==> Installing FFmpeg via Chocolatey"
            choco install ffmpeg -y
        } else {
            Write-Error "ffmpeg/ffprobe not found on PATH and neither winget nor choco is available. Install FFmpeg and re-run."
            exit 1
        }
    }
}

$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    $PyCmd = $null
    $PyArgs = @()
    foreach ($candidate in @(
        @("py", "-3.12"),
        @("py", "-3.11"),
        @("py", "-3.10"),
        @("py", "-3"),
        @("python")
    )) {
        $exe = $candidate[0]
        if (Get-Command $exe -ErrorAction SilentlyContinue) {
            $argsCheck = @()
            if ($candidate.Length -gt 1) { $argsCheck += $candidate[1] }
            $argsCheck += @("-c", "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)")
            & $exe @argsCheck 2>$null
            if ($LASTEXITCODE -eq 0) {
                $PyCmd = $exe
                if ($candidate.Length -gt 1) { $PyArgs = @($candidate[1]) }
                break
            }
        }
    }
    if (-not $PyCmd) {
        Write-Error "Python >=3.10 not found. Install Python 3.12 from python.org or Microsoft Store and retry."
        exit 1
    }
    Write-Host "==> Creating virtual environment at $VenvDir"
    & $PyCmd @PyArgs -m venv $VenvDir
}

Write-Host "==> Upgrading pip, setuptools, and wheel"
& $VenvPy -m pip install --upgrade pip setuptools wheel

$Extras = @()
if (-not $NoDev) { $Extras += "dev" }
if ($WithAntigravity) { $Extras += "antigravity" }

$InstallTarget = if ($Extras.Count -gt 0) { ".[$($Extras -join ',')]" } else { "." }
Write-Host "==> Installing video-editing-ai ($InstallTarget)"
if ($Editable) {
    & $VenvPy -m pip install -e $InstallTarget
} else {
    & $VenvPy -m pip install $InstallTarget
}

if ($SyncSkill -or -not $NoSyncSkill) {
    Write-Host "==> Synchronizing ai-video-editor skill globally (~/.gemini/config/skills)"
    & $VenvPy -m video_editing_ai.cli sync-skill
}

Write-Host "==> Verifying environment with ai-editor doctor"
& $VenvPy -m video_editing_ai.cli doctor
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "==> Setup complete!"
Write-Host "  - Primary Usage (Antigravity): Open Antigravity in this repo (or any folder) and ask to edit any media folder into a 4K HDR film."
Write-Host "  - Optional CLI / Web Studio: .\$VenvDir\Scripts\Activate.ps1 ; ai-editor --help"
