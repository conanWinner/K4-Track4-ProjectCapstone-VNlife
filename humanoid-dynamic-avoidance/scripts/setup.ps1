# Setup for humanoid-dynamic-avoidance (Windows PowerShell 5.1+).
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 [-Python python] [-SkipPip]
# 1) create .venv if missing (with system site-packages, so a preinstalled torch is reused)
# 2) pip install -r requirements.txt
# 3) clone unitree_rl_gym into third_party/ at the SHA pinned in configs/upstream_version.txt
# 4) verify the pretrained G1 policy file exists
param(
    [string]$Python = "python",
    [switch]$SkipPip
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# ---- 1. venv
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Host "[setup] creating .venv with $Python"
    & $Python -m venv --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
} else {
    Write-Host "[setup] .venv exists"
}

# ---- 2. requirements
if (-not $SkipPip) {
    Write-Host "[setup] pip install -r requirements.txt"
    & $VenvPy -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

# ---- 3. upstream clone at pinned SHA
$PinFile = Join-Path $Root "configs\upstream_version.txt"
$Pin = @{}
foreach ($line in Get-Content $PinFile) {
    if ($line -match '^\s*([A-Za-z_]+)\s*:\s*(.+?)\s*$') { $Pin[$Matches[1]] = $Matches[2] }
}
$Repo = $Pin["repo"]; $Sha = $Pin["sha"]
$Up = Join-Path $Root "third_party\unitree_rl_gym"
if (-not (Test-Path (Join-Path $Up ".git"))) {
    Write-Host "[setup] cloning $Repo @ $Sha"
    New-Item -ItemType Directory -Force (Join-Path $Root "third_party") | Out-Null
    git init -q $Up
    git -C $Up remote add origin $Repo
    git -C $Up fetch -q --depth 1 origin $Sha
    if ($LASTEXITCODE -ne 0) { throw "git fetch of pinned SHA failed (network?)" }
    git -C $Up checkout -q FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed" }
} else {
    $Head = (git -C $Up rev-parse HEAD).Trim()
    if ($Head -ne $Sha) {
        Write-Warning "[setup] third_party/unitree_rl_gym is at $Head, pinned SHA is $Sha (not changing it)"
    } else {
        Write-Host "[setup] unitree_rl_gym at pinned SHA $Sha"
    }
}

# ---- 4. policy + scene check
$Policy = Join-Path $Up $Pin["policy"]
$Scene = Join-Path $Up $Pin["scene_xml"]
if (-not (Test-Path $Policy)) {
    Write-Error ("[setup] Pretrained G1 policy NOT found at $Policy. It is shipped by unitree_rl_gym " +
        "(deploy/pre_train/g1/motion.pt); re-clone third_party/unitree_rl_gym. The G1 path cannot run without it " +
        "(the AMR proxy robot still works).")
    exit 1
}
if (-not (Test-Path $Scene)) { Write-Error "[setup] G1 scene xml not found at $Scene"; exit 1 }
Write-Host "[setup] policy OK: $Policy"

& $VenvPy -c "import mujoco, torch; print('[setup] mujoco', mujoco.__version__, '| torch', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "mujoco/torch import failed in .venv" }
Write-Host "[setup] done. Quick check: .venv\Scripts\python.exe -m pytest tests\test_g1_adapter.py -q"
