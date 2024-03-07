import platform

if platform.system() == "Darwin":
    version = tuple(map(int, platform.mac_ver()[0].split(".")))
    major, minor = version[0], version[1]
    if (major, minor) < (13, 5):
        raise ImportError(
            f"Only macOS 13.5 and newer are supported, not {major}.{minor}"
        )
