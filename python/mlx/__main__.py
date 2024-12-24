import argparse


def main() -> None:
    from mlx.core import __version__

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print the version number.",
    )
    parser.add_argument(
        "--cmake-dir",
        action="store_true",
        help="Print the path to the MLX CMake module directory.",
    )
    args = parser.parse_args()
    if args.cmake_dir:
        from pathlib import Path

        print(Path(__file__).parent)


if __name__ == "__main__":
    main()
