"""XPU Triton environment setup for Windows + oneAPI (path auto-detection).

Import this module (or call setup()) BEFORE any triton/fla code to set up:
- ONEAPI_ROOT, LEVEL_ZERO_V1_SDK_PATH (Triton's find_sycl)
- CC/CXX (oneAPI clang-cl with full path)
- INCLUDE/LIB (MSVC C++ std lib + Win SDK, Triton doesn't add these itself)
- os.add_dll_directory + pre-load sycl8.dll (CRITICAL: must happen before
  any Triton code runs, or the SYCL runtime initializes in host-only mode
  and the launcher's device code registration fails with WinError 127)

The toolchain is auto-detected so the module runs on machines other than the
one it was developed on: oneAPI and Windows SDK installs are globbed for the
newest version, the MSVC toolset is located through vswhere (falling back to
a directory glob), and anything that cannot be found degrades to a loud
warning instead of a crash. Set ONEAPI_ROOT, LEVEL_ZERO_V1_SDK_PATH, CC, CXX,
INCLUDE or LIB yourself to override a probe.

Also patches triton's build.py and driver.py (idempotent, source-pattern
matched so a different triton version is left alone — but loudly):
- -fsycl for every XPU compilation unit (clang-cl path)
- winmode=0 + os.add_dll_directory in SpirvUtils/TritonLauncher
- separating compiler flags from linker flags in MSVC mode

report() prints what was found/patched so a broken toolchain fails fast,
not silently. kev.serve calls report() once when it starts.
"""
import ctypes, importlib.metadata, os, pathlib, subprocess

_PAGE = []            # (what, detail) — everything probed/patched, for report()
_setup_done = False

def _log(kind, detail):
    _PAGE.append((kind, detail))

def _newest_child(parent):
    p = pathlib.Path(parent)
    if not p.is_dir():
        return None
    try:
        vers = sorted(p.iterdir(), key=lambda d: d.name.lower())
    except OSError:
        return None
    return vers[-1] if vers else None


def _oneapi():
    """Locate the oneAPI base, its compiler bin (clang-cl + sycl8.dll) and the SDK version."""
    base = None
    env = os.environ.get("ONEAPI_ROOT", "").strip()
    if env and pathlib.Path(env).is_dir():
        base = pathlib.Path(env)
    else:
        for cand in (r"C:\Program Files (x86)\Intel\oneAPI", r"C:\Program Files\Intel\oneAPI",
                     os.path.expanduser(r"~\intel\oneapi"), r"C:\Intel\oneAPI"):
            if pathlib.Path(cand).is_dir():
                base = pathlib.Path(cand)
                break
    if base is None:
        _log("warn", "ONEAPI_ROOT not found (set ONEAPI_ROOT to the oneAPI base directory)")
        return None
    _log("oneapi", str(base))
    # clang-cl and sycl8.dll live in different oneAPI layouts: the dll under
    # <base>/compiler/{latest,<ver>}/bin, the compiler under those bins or <base>/<ver>/bin/compiler.
    comp = base / "compiler"
    vers = sorted([p for p in comp.iterdir() if p.is_dir() and p.name != "latest"]) if comp.is_dir() else []
    bases_ver = sorted([p for p in base.iterdir() if p.is_dir() and p.name not in ("compiler", "common")]) if base.is_dir() else []
    clang = next((c for cand in [comp / "latest" / "bin", comp / "latest" / "bin" / "compiler",
                                 *[v / "bin" for v in vers], *[v / "bin" / "compiler" for v in vers],
                                 *[v / "bin" / "compiler" for v in bases_ver], *[v / "bin" for v in bases_ver]]
                  for c in [cand] if (cand / "clang-cl.exe").is_file()), None)
    dllbin = next((c for cand in [comp / "latest" / "bin", *[v / "bin" for v in vers]]
                   for c in [cand] if (cand / "sycl8.dll").is_file()), None)
    if clang is None or dllbin is None:
        _log("warn", f"oneAPI compiler not fully found under {base} (need clang-cl.exe and sycl8.dll; set ONEAPI_ROOT and CC/CXX manually)")
    else:
        _log("clang-cl", str(clang))
        _log("sycl8", str(dllbin / "sycl8.dll"))
    return base, clang, dllbin


def _msvc():
    """MSVC include/lib dirs via vswhere, else a directory scan."""
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
        except Exception as e:
            _log("warn", f"vswhere failed: {e}")
    msvc_root = (install / "VC" / "Tools" / "MSVC") if install else None
    if msvc_root is None or not msvc_root.is_dir():
        for base in (r"C:\Program Files (x86)\Microsoft Visual Studio", r"C:\Program Files\Microsoft Visual Studio"):
            hit = _newest_child(pathlib.Path(base) / "VC" / "Tools" / "MSVC") if pathlib.Path(base).exists() else None
            if hit is not None and (hit / "include").is_dir():
                msvc_root = hit.parent
                break
    if msvc_root is None or not msvc_root.is_dir():
        _log("warn", "MSVC VC tools not found (install the 'Desktop development with C++' workload or set INCLUDE/LIB manually)")
        return None, None
    ver = _newest_child(msvc_root)
    if ver is None or not (ver / "include").is_dir():
        _log("warn", f"no MSVC version dir under {msvc_root}")
        return None, None
    _log("msvc", str(ver))
    return str(ver / "include"), str(ver / "lib" / "x64")


def _winsdk():
    """Win SDK include/lib versioned roots (ucrt/um/shared); None when absent."""
    root = pathlib.Path(r"C:\Program Files (x86)\Windows Kits\10")
    inc_root, lib_root = root / "Include", root / "Lib"
    if not (inc_root.is_dir() and lib_root.is_dir()):
        _log("warn", "Windows 10 SDK not found under %s" % root)
        return None
    ver = _newest_child(inc_root)
    if ver is None or not (lib_root / ver.name).is_dir():
        _log("warn", "Windows 10 SDK versions inconsistent under %s" % root)
        return None
    _log("winsdk", str(ver))
    return root, ver.name


def _triton_version():
    """The installed triton distribution's version (triton / pytorch-triton-xpu / intel-triton)."""
    for dist in ("triton", "pytorch-triton-xpu", "intel-triton"):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _patch_triton():
    """Apply the three triton patches, reporting each as applied/skipped. Patterns are matched
    verbatim against the installed source, so a different triton version simply reports skipped."""
    tv = _triton_version()
    _log("triton", tv)
    if tv not in ("3.2", "3.2.0", "3.3", "3.3.0", "3.4", "3.4.0"):
        _log("warn", f"triton {tv}: patches were written against 3.4; a mismatch reports skipped, not applied")

    import sysconfig
    venv = pathlib.Path(sysconfig.get_paths()["purelib"])
    driver_py = venv / "triton" / "backends" / "intel" / "driver.py"
    build_py = venv / "triton" / "runtime" / "build.py"
    ds = driver_py.read_text(encoding="utf-8") if driver_py.is_file() else ""
    bs = build_py.read_text(encoding="utf-8") if build_py.is_file() else ""
    # The Intel wheel (pytorch-triton-xpu) ships all of this natively: sycl8 preload + winmode=0 in
    # both DLL launchers, the -fsycl flags for every compile unit, and the MSVC compiler/linker flag
    # split in build.py. When present, nothing is written to site-packages — this is the normal case.
    native = "winmode=0" in ds and "sycl8.dll" in ds and "spirv_utils" in ds and '"/link" in cc_cmd' in bs
    if native:
        _log("patches", "none needed (pytorch-triton-xpu already loads sycl8, uses winmode=0 and splits MSVC flags)")
        return
    # Stock triton (or an older Intel wheel): apply the source patches, each reported applied/skipped.
    _patch_file(build_py, "build.py (compiler/linker flag split)",
        '    cc_cmd = _cc_cmd(cc, src, so, include_dirs, library_dirs, libraries)\n    cc_cmd += extra_compile_args',
        '    cc_cmd = _cc_cmd(cc, src, so, include_dirs, library_dirs, libraries)\n'
        '    if "clang-cl" in cc and "/link" in cc_cmd:\n'
        '        link_idx = cc_cmd.index("/link")\n'
        '        pre_link, post_link = [], []\n'
        '        for arg in extra_compile_args:\n'
        '            if arg.startswith("/LIBPATH") or arg.startswith("/L") or arg.endswith(".lib"):\n'
        '                post_link.append(arg)\n'
        '            else:\n'
        '                pre_link.append(arg)\n'
        '        cc_cmd = cc_cmd[:link_idx] + pre_link + cc_cmd[link_idx:]\n'
        '        cc_cmd += post_link\n'
        '    else:\n'
        '        cc_cmd += extra_compile_args')
    driver_py = venv / "triton" / "backends" / "intel" / "driver.py"
    _patch_file(driver_py, "driver.py (-fsycl for all XPU compiles)",
        'if name == "__triton_launcher" and os.name == "nt":\n                extra_compiler_args += ["-fsycl"]',
        'if name in ("__triton_launcher", "spirv_utils", "arch_utils") and os.name == "nt":\n                extra_compiler_args += ["-fsycl"]')
    _patch_file(driver_py, "driver.py (SpirvUtils winmode=0)",
        'class SpirvUtils:\n\n    def __init__(self, cache_path: str):\n        self.shared_library = ctypes.PyDLL(cache_path)',
        'class SpirvUtils:\n\n    def __init__(self, cache_path: str):\n        _ob = os.path.join(os.environ.get("ONEAPI_ROOT", ""), "compiler", "latest", "bin")\n        if os.path.isdir(_ob): os.add_dll_directory(_ob)\n        self.shared_library = ctypes.PyDLL(cache_path, winmode=0)')
    _patch_file(driver_py, "driver.py (TritonLauncher sycl8 preload)",
        'class TritonLauncher:\n\n    def __init__(self, cache_path: str):\n        self.shared_library = ctypes.PyDLL(cache_path)',
        'class TritonLauncher:\n\n    def __init__(self, cache_path: str):\n        _ob = os.path.join(os.environ.get("ONEAPI_ROOT", ""), "compiler", "latest", "bin")\n        if os.path.isdir(_ob):\n            os.add_dll_directory(_ob)\n            ctypes.WinDLL(os.path.join(_ob, "sycl8.dll"), winmode=0)\n        self.shared_library = ctypes.PyDLL(cache_path, winmode=0)')


def _patch_file(path, what, old, new):
    """Idempotent source patch: apply when `old` matches and `new` isn't already there."""
    p = pathlib.Path(path)
    if not p.is_file():
        _log("patch", f"{what}: SKIPPED ({p} missing — is triton intel backend installed?)")
        return
    src = p.read_text(encoding="utf-8")
    if new in src:
        _log("patch", f"{what}: already applied")
        return
    if old in src:
        p.write_text(src.replace(old, new), encoding="utf-8")
        _log("patch", f"{what}: applied")
    else:
        _log("patch", f"{what}: SKIPPED (triton {_triton_version()} source does not match 3.4 patterns)")


def setup():
    """Probe the toolchain and patch triton. Call before importing triton/fla."""
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    base, clang, dllbin = _oneapi()
    if base is not None:
        os.environ.setdefault("ONEAPI_ROOT", str(base))
    # LevelZero SDK (for triton's find_sycl / default PATH search); a miss is not fatal.
    lvl = os.environ.get("LEVEL_ZERO_V1_SDK_PATH", "")
    if not lvl or not pathlib.Path(lvl).is_dir():
        hit = _newest_child(pathlib.Path(r"C:\Program Files\LevelZeroSDK"))
        if hit is not None:
            os.environ["LEVEL_ZERO_V1_SDK_PATH"] = str(hit)
            _log("levelzero", str(hit))
        else:
            _log("warn", "LevelZero SDK not found (triton may still work with the level-zero-loader DLL on PATH)")
    if dllbin is not None:
        os.add_dll_directory(str(dllbin))
        try:
            ctypes.WinDLL(os.path.join(dllbin, "sycl8.dll"), winmode=0)
            _log("sycl8", "preloaded from " + str(dllbin / "sycl8.dll"))
        except OSError as e:
            _log("warn", f"sycl8.dll preload failed: {e}")
    if clang is not None:
        os.environ.setdefault("CC", str(clang))
        os.environ.setdefault("CXX", str(clang))
    msvc_inc, msvc_lib = _msvc()
    winsdk_root, winsdk_ver = _winsdk()
    if msvc_inc is not None:
        inc = [msvc_inc]
        if winsdk_ver is not None:
            inc += [str(winsdk_root / "Include" / winsdk_ver / d) for d in ("ucrt", "um", "shared")]
        os.environ["INCLUDE"] = ";".join(inc + [os.environ.get("INCLUDE", "")])
    if msvc_lib is not None:
        lib = [msvc_lib]
        if winsdk_ver is not None:
            lib += [str(winsdk_root / "Lib" / winsdk_ver / d / "x64") for d in ("ucrt", "um")]
        os.environ["LIB"] = ";".join(lib + [os.environ.get("LIB", "")])
    _patch_triton()


def report():
    """One line per probe/patch result, for the server's startup banner and diagnostics."""
    return "\n".join(f"{kind}: {detail}" for kind, detail in _PAGE)


setup()