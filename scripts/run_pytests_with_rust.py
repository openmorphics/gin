#!/usr/bin/env python3
"""
Run pytest for the Python package under tonic-py/ with Rust fast paths enabled.

Usage:
  python tonic-rs/scripts/run_pytests_with_rust.py [--ops voxel_grid,to_frame] --pytest [pytest args...]

Examples:
  python tonic-rs/scripts/run_pytests_with_rust.py --pytest -q
  python tonic-rs/scripts/run_pytests_with_rust.py --ops voxel_grid,to_frame --pytest -k voxel
"""

import argparse
import importlib
import os
import sys
import subprocess
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_pytests_with_rust",
        description="Run tonic-py tests with Rust fast paths enabled (TONIC_USE_RUST=1).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated subset of Rust ops to enable (sets TONIC_RUST_OPS). Example: voxel_grid,to_frame",
    )
    parser.add_argument(
        "--pytest",
        nargs=argparse.REMAINDER,
        help="Arguments to pass through to pytest (everything after --pytest is forwarded).",
        default=[],
    )
    return parser.parse_args()


def ensure_extension_importable(pyproject_path: Path) -> None:
    try:
        importlib.import_module("tonic_python")
    except Exception as e:
        print("Rust extension module 'tonic_python' is not importable in this Python environment.", file=sys.stderr)
        print("", file=sys.stderr)
        print(f"Interpreter: {sys.executable}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Build the extension locally with maturin, then re-run this script:", file=sys.stderr)
        print("  pip install maturin", file=sys.stderr)
        print(f"  maturin develop -m {pyproject_path}", file=sys.stderr)
        print("", file=sys.stderr)
        print("If you use virtualenv/conda, ensure you're in the same environment when running the command above and this script.", file=sys.stderr)
        sys.exit(2)


def main() -> int:
    args = parse_args()

    # Resolve repository paths relative to this script
    script_path = Path(__file__).resolve()
    # repo root containing 'tonic-rs/' and 'tonic-py/' (three levels up from this file)
    workspace_root = script_path.parents[2]
    py_pkg_dir = workspace_root / "tonic-py"
    tests_dir = py_pkg_dir / "test"
    pyproject_path = workspace_root / "tonic-rs" / "tonic-python" / "pyproject.toml"

    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}", file=sys.stderr)
        sys.exit(2)

    # Verify the Rust extension is importable before running tests
    ensure_extension_importable(pyproject_path)

    env = os.environ.copy()
    # Ensure PYTHONPATH includes tonic-py so 'import tonic' resolves to the in-repo package
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{py_pkg_dir}{os.pathsep}{existing_pp}" if existing_pp else str(py_pkg_dir)

    # Enable Rust fast paths by default
    env.setdefault("TONIC_USE_RUST", "1")

    if args.ops:
        env["TONIC_RUST_OPS"] = args.ops

    cmd = [sys.executable, "-m", "pytest", str(tests_dir)]
    if args.pytest:
        cmd.extend(args.pytest)

    print(f"PYTHONPATH prepended with: {py_pkg_dir}")
    print(f"TONIC_USE_RUST={env.get('TONIC_USE_RUST')} TONIC_RUST_OPS={env.get('TONIC_RUST_OPS', '(all)')}")
    print("Running:", " ".join(cmd))

    proc = subprocess.run(cmd, cwd=str(workspace_root), env=env)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())