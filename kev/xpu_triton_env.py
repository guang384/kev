"""XPU Triton environment setup for Windows + oneAPI (auto-detected toolchain).

Import this module (or call setup()) BEFORE any triton/fla code runs so that ONEAPI_ROOT /
LEVEL_ZERO_V1_SDK_PATH point at the real installs, CC/CXX are the oneAPI clang-cl, INCLUDE/LIB
carry the MSVC + Windows SDK paths triton doesn't add itself, and sycl8.dll is on the DLL path
and pre-loaded (otherwise the SYCL runtime initializes host-only and Triton fails with WinError 127).

Every path is probed (newest installed version); set ONEAPI_ROOT, LEVEL_ZERO_V1_SDK_PATH, CC, CXX,
INCLUDE or LIB yourself to override a probe. Requires the Intel wheel (pytorch-triton-xpu), whose
drivers already load sycl8 + use winmode=0 and whose build.py splits MSVC flags — no site-packages
patch. report() prints the probe results; kev.serve shows them at startup.
"""
import ctypes, os, pathlib, subprocess

_PAGE = []            # (kind, detail) — what was found, for report()
_setup_done = False


def _log(kind, detail):
    _PAGE.append((kind, detail))


def _ver_key(name):
    """A sortable key for versioned directory names: dotted numeric components compare numerically
    (2025.10 > 2025.2, 14.40 > 14.9), and non-numeric ones ('latest') sort after all real versions."""
    return tuple((1, int(part)) if part.isdigit() else (2, part) for part in str(name).split("."))


def _newest(parent):
    """The newest directory under `parent` (by numeric version, else name), or None."""
    p = pathlib.Path(parent)
    if not p.is_dir():
        return None
    try:
        vers = [d for d in p.iterdir() if d.is_dir()]
    except OSError:
        return None
    return max(vers, key=lambda d: _ver_key(d.name)) if vers else None


def _oneapi():
    """-> (base, clang-cl exe, sycl bin dir) or (None, None, None)."""
    env = os.environ.get("ONEAPI_ROOT", "").strip()
    base = pathlib.Path(env) if env and pathlib.Path(env).is_dir() else None
    if base is None:
        for cand in (r"C:\Program Files (x86)\Intel\oneAPI", r"C:\Program Files\Intel\oneAPI",
                     os.path.expanduser(r"~\intel\oneapi")):
            if pathlib.Path(cand).is_dir():
                base = pathlib.Path(cand)
                break
    if base is None:
        _log("warn", "ONEAPI_ROOT not found; set ONEAPI_ROOT to the oneAPI base dir")
        return None, None, None
    _log("oneapi", str(base))
    comp = base / "compiler"
    dll = comp / "latest" / "bin"
    if not (dll / "sycl8.dll").is_file():                       # older layouts
        newest = _newest(comp)
        dll = newest / "bin" if newest else None
    clang = None
    for cand in (dll / "compiler" / "clang-cl.exe", dll / "clang-cl.exe") if dll else ():
        if cand.is_file():
            clang = cand
            break
    if clang is None:
        for cand in (comp / "latest" / "bin" / "compiler" / "clang-cl.exe",):
            if cand.is_file():
                clang = cand
                break
    if clang is None or dll is None:
        _log("warn", f"no clang-cl.exe / sycl8.dll under {base}; set ONEAPI_ROOT and CC/CXX")
        return base, None, None
    _log("clang-cl", str(clang))
    return base, clang, dll


def _msvc():
    """MSVC include/lib dirs via vswhere, else the newest installed toolset."""
    vswhere = pathlib.Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    install = None
    if vswhere.exists():
        try:
            out = subprocess.run([str(vswhere), "-latest", "-products", "*",
                                  "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                  "-property", "installationPath"],
                                 capture_output=True, text=True, timeout=20)
            if out.returncode == 0 and out.stdout.strip():
                install = pathlib.Path(out.stdout.strip())
        except Exception:
            pass
    root = install and _newest(install / "VC" / "Tools" / "MSVC") or None
    if root is None:
        for base in (r"C:\Program Files (x86)\Microsoft Visual Studio", r"C:\Program Files\Microsoft Visual Studio"):
            if (hit := _newest(pathlib.Path(base) / "VC" / "Tools" / "MSVC")):
                root = hit
                break
    if root is None or not (root / "include").is_dir():
        _log("warn", "MSVC VC tools not found; install the C++ workload or set INCLUDE/LIB")
        return None, None
    _log("msvc", str(root))
    return str(root / "include"), str(root / "lib" / "x64")


def _winsdk():
    """The newest Windows 10 SDK version root, or None."""
    root = pathlib.Path(r"C:\Program Files (x86)\Windows Kits\10")
    ver = _newest(root / "Include") if (root / "Include").is_dir() else None
    if ver is None or not (root / "Lib" / ver.name).is_dir():
        _log("warn", "Windows SDK not found under %s" % root)
        return None
    _log("winsdk", str(ver))
    return ver.name


def setup():
    """Probe the toolchain and prepare the environment. Call before importing torch/triton/fla:
    the preload can only claim the sycl8.dll module name before torch loads its wheel copy, and a
    process that missed that window must instead keep the wheel's sycl8 end to end (its launchers
    then build against the wheel's import library and match at load time)."""
    global _setup_done
    if _setup_done:
        return
    _setup_done = True
    late = False
    if os.name == "nt":
        ctypes.windll.kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        ctypes.windll.kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        late = bool(ctypes.windll.kernel32.GetModuleHandleW("sycl8.dll"))   # torch already claimed it
    if late:
        _log("sycl8", "already loaded by torch; skipping the oneAPI preload (call setup() before importing torch)")
        return
    # A preloaded process compiles/links against the oneAPI sycl8, so its triton cache must not be
    # shared with cold processes (which resolve the wheel's sycl8): a launcher built in one fails
    # to load in the other (0xc0000139, a modal Entry-Point dialog). Set your own to override.
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".triton", "cache-xpu-oneapi"))
    base, clang, dll = _oneapi()
    if base is not None:
        os.environ.setdefault("ONEAPI_ROOT", str(base))
    lvl = os.environ.get("LEVEL_ZERO_V1_SDK_PATH", "")
    if not lvl or not pathlib.Path(lvl).is_dir():
        hit = _newest(pathlib.Path(r"C:\Program Files\LevelZeroSDK"))
        if hit is not None:
            os.environ["LEVEL_ZERO_V1_SDK_PATH"] = str(hit)
            _log("levelzero", str(hit))
    if dll is not None:
        os.add_dll_directory(str(dll))
        try:
            ctypes.WinDLL(os.path.join(dll, "sycl8.dll"), winmode=0)
            _log("sycl8", "preloaded")
        except OSError as e:
            _log("warn", f"sycl8.dll preload failed: {e}")
    if clang is not None:
        os.environ.setdefault("CC", str(clang))
        os.environ.setdefault("CXX", str(clang))
    inc, lib = _msvc()
    sdk = _winsdk()
    if inc is not None:
        parts = [inc]
        if sdk is not None:
            parts += [str(pathlib.Path(r"C:\Program Files (x86)\Windows Kits\10") / "Include" / sdk / d)
                      for d in ("ucrt", "um", "shared")]
        os.environ["INCLUDE"] = ";".join(parts + [os.environ.get("INCLUDE", "")])
    if lib is not None:
        parts = [lib]
        if sdk is not None:
            parts += [str(pathlib.Path(r"C:\Program Files (x86)\Windows Kits\10") / "Lib" / sdk / d / "x64")
                      for d in ("ucrt", "um")]
        os.environ["LIB"] = ";".join(parts + [os.environ.get("LIB", "")])


def report():
    """One line per probe result, for kev.serve's startup banner and diagnostics."""
    return "\n".join(f"{k}: {v}" for k, v in _PAGE)


setup()