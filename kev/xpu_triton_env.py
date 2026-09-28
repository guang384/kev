"""XPU Triton environment setup for Windows + oneAPI.

Import this module (or call setup()) BEFORE any triton/fla code to set up:
- ONEAPI_ROOT, LEVEL_ZERO_V1_SDK_PATH (Triton's find_sycl)
- CC/CXX (oneAPI clang-cl with full path)
- INCLUDE/LIB (MSVC C++ std lib + Win SDK, Triton doesn't add these itself)
- os.add_dll_directory + pre-load sycl8.dll (CRITICAL: must happen before
  any Triton code runs, or the SYCL runtime initializes in host-only mode
  and the launcher's device code registration fails with WinError 127)

Also patches Triton's build.py and driver.py at import time for:
- -fsycl flag for all XPU compilations (clang-cl path)
- winmode=0 + os.add_dll_directory in SpirvUtils/TritonLauncher
- Separating compiler flags from linker flags in MSVC mode
"""
import os, sys, pathlib, ctypes

_setup_done = False

def setup():
    """Set up the XPU Triton environment. Call before importing triton/fla."""
    global _setup_done
    if _setup_done: return
    _setup_done = True

    os.environ["ONEAPI_ROOT"] = r"C:\Program Files (x86)\Intel\oneAPI"
    os.environ["LEVEL_ZERO_V1_SDK_PATH"] = r"C:\Program Files\LevelZeroSDK\1.26.1"

    _ONEAPI_BIN = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin"
    os.add_dll_directory(_ONEAPI_BIN)

    # CRITICAL: pre-load sycl8.dll before any Triton code runs
    ctypes.WinDLL(os.path.join(_ONEAPI_BIN, "sycl8.dll"), winmode=0)

    os.environ["CC"] = r"C:\Program Files (x86)\Intel\oneAPI\2025.3\bin\compiler\clang-cl.exe"
    os.environ["CXX"] = r"C:\Program Files (x86)\Intel\oneAPI\2025.3\bin\compiler\clang-cl.exe"

    _msvc_inc = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\include"
    _winsdk = r"C:\Program Files (x86)\Windows Kits\10"
    _winsdk_ver = sorted(pathlib.Path(_winsdk, "Include").iterdir())[-1].name
    os.environ["INCLUDE"] = ";".join([
        _msvc_inc,
        str(pathlib.Path(_winsdk, "Include", _winsdk_ver, "ucrt")),
        str(pathlib.Path(_winsdk, "Include", _winsdk_ver, "um")),
        str(pathlib.Path(_winsdk, "Include", _winsdk_ver, "shared")),
        os.environ.get("INCLUDE", ""),
    ])
    _winsdk_lib_ver = sorted(pathlib.Path(_winsdk, "Lib").iterdir())[-1].name
    os.environ["LIB"] = ";".join([
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\lib\x64",
        str(pathlib.Path(_winsdk, "Lib", _winsdk_lib_ver, "ucrt", "x64")),
        str(pathlib.Path(_winsdk, "Lib", _winsdk_lib_ver, "um", "x64")),
        os.environ.get("LIB", ""),
    ])
    _patch_triton()

# --- Triton patches (applied at import time, before triton is imported) ---

def _patch_triton():
    import importlib, sysconfig
    venv = pathlib.Path(sysconfig.get_paths()["purelib"])
    
    # Patch build.py: separate compiler/linker flags in MSVC mode
    build_py = venv / "triton" / "runtime" / "build.py"
    if build_py.exists():
        src = build_py.read_text(encoding="utf-8")
        old = '    cc_cmd = _cc_cmd(cc, src, so, include_dirs, library_dirs, libraries)\n    cc_cmd += extra_compile_args'
        new = '''    cc_cmd = _cc_cmd(cc, src, so, include_dirs, library_dirs, libraries)
    if "clang-cl" in cc and "/link" in cc_cmd:
        link_idx = cc_cmd.index("/link")
        pre_link, post_link = [], []
        for arg in extra_compile_args:
            if arg.startswith("/LIBPATH") or arg.startswith("/L") or arg.endswith(".lib"):
                post_link.append(arg)
            else:
                pre_link.append(arg)
        cc_cmd = cc_cmd[:link_idx] + pre_link + cc_cmd[link_idx:]
        cc_cmd += post_link
    else:
        cc_cmd += extra_compile_args'''
        if old in src:
            src = src.replace(old, new)
            build_py.write_text(src, encoding="utf-8")

    # Patch driver.py: -fsycl for all XPU compilations + winmode=0 + DLL dirs
    driver_py = venv / "triton" / "backends" / "intel" / "driver.py"
    if driver_py.exists():
        src = driver_py.read_text(encoding="utf-8")
        # Add -fsycl for all compilations on Windows
        src = src.replace(
            'if name == "__triton_launcher" and os.name == "nt":\n                extra_compiler_args += ["-fsycl"]',
            'if name in ("__triton_launcher", "spirv_utils", "arch_utils") and os.name == "nt":\n                extra_compiler_args += ["-fsycl"]'
        )
        # Patch SpirvUtils to use winmode=0 + DLL dir
        src = src.replace(
            'class SpirvUtils:\n\n    def __init__(self, cache_path: str):\n        self.shared_library = ctypes.PyDLL(cache_path)',
            'class SpirvUtils:\n\n    def __init__(self, cache_path: str):\n        _ob = os.path.join(os.environ.get("ONEAPI_ROOT", ""), "compiler", "latest", "bin")\n        if os.path.isdir(_ob): os.add_dll_directory(_ob)\n        self.shared_library = ctypes.PyDLL(cache_path, winmode=0)'
        )
        # Patch TritonLauncher to pre-load sycl8 + winmode=0 + DLL dir
        src = src.replace(
            'class TritonLauncher:\n\n    def __init__(self, cache_path: str):\n        self.shared_library = ctypes.PyDLL(cache_path)',
            'class TritonLauncher:\n\n    def __init__(self, cache_path: str):\n        _ob = os.path.join(os.environ.get("ONEAPI_ROOT", ""), "compiler", "latest", "bin")\n        if os.path.isdir(_ob):\n            os.add_dll_directory(_ob)\n            ctypes.WinDLL(os.path.join(_ob, "sycl8.dll"), winmode=0)\n        self.shared_library = ctypes.PyDLL(cache_path, winmode=0)'
        )
        driver_py.write_text(src, encoding="utf-8")

# Run setup at import time (after _patch_triton is defined)
setup()
