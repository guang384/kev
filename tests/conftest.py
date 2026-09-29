"""Runs before any test module (and so before torch) is imported: on Windows this preloads the
oneAPI sycl8 and the toolchain into the process. Without it torch loads the intel-sycl-rt wheel's
sycl8.dll, and the first Triton XPU JIT (any fla op's autotune reaches it even on CPU tensors)
fails resolving __triton_launcher's symbols — Windows shows a modal Entry-Point dialog (0xc0000139)
that hangs the run. kev.serve does the same preload before torch (see kev/xpu_triton_env.py)."""
import os
import sys

if sys.platform == "win32" and os.environ.get("KEV_XPU_TRITON", "1") != "0":
    from kev import xpu_triton_env  # noqa: F401