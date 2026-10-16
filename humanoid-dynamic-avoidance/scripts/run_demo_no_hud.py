"""Run the demo without the unstable MuJoCo text HUD on Linux."""
import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")

from mujoco import viewer
from run_demo import main

# ponytail: omit text HUD; return to run_demo.py when MuJoCo HUD works on this host.
viewer.Handle.set_texts = lambda self, texts: None

if __name__ == "__main__":
    raise SystemExit(main())
