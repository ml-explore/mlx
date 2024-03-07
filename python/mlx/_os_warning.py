import platform

if platform.system() == "Darwin":
    major, minor, _ = map(int, platform.mac_ver()[0].split("."))
    if (major, minor) < (13, 5):
        raise OSError(f"Only macOS 13.5 and newer are supported, not {major}.{minor}")
