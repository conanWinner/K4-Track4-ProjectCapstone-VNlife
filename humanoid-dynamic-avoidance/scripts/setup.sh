#!/usr/bin/env bash
# Setup for humanoid-dynamic-avoidance (Linux/macOS/Git Bash).
#   bash scripts/setup.sh            (PYTHON=python3.11 bash scripts/setup.sh to pick an interpreter)
#   SKIP_PIP=1 bash scripts/setup.sh to skip pip
# 1) create .venv if missing  2) pip install -r requirements.txt
# 3) clone unitree_rl_gym at the SHA pinned in configs/upstream_version.txt  4) verify policy file
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

if [ -x .venv/Scripts/python.exe ]; then VENV_PY=.venv/Scripts/python.exe;
elif [ -x .venv/bin/python ]; then VENV_PY=.venv/bin/python;
else
  echo "[setup] creating .venv with $PYTHON"
  "$PYTHON" -m venv --system-site-packages .venv
  if [ -x .venv/Scripts/python.exe ]; then VENV_PY=.venv/Scripts/python.exe; else VENV_PY=.venv/bin/python; fi
fi

if [ -z "${SKIP_PIP:-}" ]; then
  echo "[setup] pip install -r requirements.txt"
  "$VENV_PY" -m pip install -r requirements.txt
fi

pin() { sed -n "s/^$1:[[:space:]]*//p" configs/upstream_version.txt | tr -d '\r' | head -1; }
REPO="$(pin repo)"; SHA="$(pin sha)"
UP="third_party/unitree_rl_gym"
if [ ! -d "$UP/.git" ]; then
  echo "[setup] cloning $REPO @ $SHA"
  mkdir -p third_party
  git init -q "$UP"
  git -C "$UP" remote add origin "$REPO"
  git -C "$UP" fetch -q --depth 1 origin "$SHA"
  git -C "$UP" checkout -q FETCH_HEAD
else
  HEAD="$(git -C "$UP" rev-parse HEAD)"
  if [ "$HEAD" != "$SHA" ]; then
    echo "[setup] WARNING: $UP is at $HEAD, pinned SHA is $SHA (not changing it)" >&2
  else
    echo "[setup] unitree_rl_gym at pinned SHA $SHA"
  fi
fi

POLICY="$UP/$(pin policy)"; SCENE="$UP/$(pin scene_xml)"
if [ ! -f "$POLICY" ]; then
  echo "[setup] ERROR: pretrained G1 policy NOT found at $POLICY." >&2
  echo "        It ships with unitree_rl_gym (deploy/pre_train/g1/motion.pt); re-clone $UP." >&2
  echo "        The G1 path cannot run without it (the AMR proxy robot still works)." >&2
  exit 1
fi
[ -f "$SCENE" ] || { echo "[setup] ERROR: G1 scene xml not found at $SCENE" >&2; exit 1; }
echo "[setup] policy OK: $POLICY"
"$VENV_PY" -c "import mujoco, torch; print('[setup] mujoco', mujoco.__version__, '| torch', torch.__version__)"
echo "[setup] done. Quick check: $VENV_PY -m pytest tests/test_g1_adapter.py -q"
