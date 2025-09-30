# Tonic Rust Re-write

This workspace contains a Rust re-write of Tonic kernels plus Python bindings for opt-in acceleration from the existing Python package.

## Phase 1 verification — run Python tests with Rust fast paths

Objective: Provide a reproducible way to run the existing tonic-py test suite with the Rust kernels enabled via environment variables, without modifying tests. Use the runner at [scripts/run_pytests_with_rust.py](scripts/run_pytests_with_rust.py) to set up the environment and invoke pytest.

### 1) Build the Python extension locally (once per Python environment)

The Rust-backed Python module is built and installed into your current Python environment with maturin.

- Ensure you’re in the same venv/Conda env you’ll run tests from
- Install maturin and build/install the extension

```bash
python -m pip install --upgrade pip
python -m pip install maturin
maturin develop -m tonic-python/pyproject.toml
```

If you change Rust code, re-run the maturin command above to rebuild.

### 2) Run the existing tonic-py test suite with Rust enabled

Use the runner which:
- Prepends PYTHONPATH with the in-repo tonic-py package
- Sets TONIC_USE_RUST=1 by default
- Optionally constrains enabled ops via TONIC_RUST_OPS
- Forwards any extra arguments directly to pytest

Examples:
```bash
# Run the full test suite (quiet)
python scripts/run_pytests_with_rust.py --pytest -q

# Filter tests (pytest -k)
python scripts/run_pytests_with_rust.py --pytest -k voxel

# Enable only a subset of Rust ops
python scripts/run_pytests_with_rust.py --ops voxel_grid,to_frame --pytest -k "voxel or frame"

# Show help
python scripts/run_pytests_with_rust.py -h
```

If the extension is not importable, the runner will print actionable guidance and exit early (see Troubleshooting below). Build the extension, then re-run the command.

### 3) Environment behavior

- Runner sets only child-process env vars; nothing persists after it exits.
- Defaults:
  - TONIC_USE_RUST=1 (use Rust fast paths when available)
  - TONIC_RUST_OPS unset (all supported ops enabled)
- You can restrict fast paths:
  - `--ops voxel_grid,to_frame` (equivalent to setting TONIC_RUST_OPS="voxel_grid,to_frame")
- The runner ensures tests import the in-repo Python package by prepending `tonic-py/` to PYTHONPATH, then executes pytest against `tonic-py/test/`.

### 4) Data contracts preserved by Rust paths (Phase 1)

- Voxel grid: dtype float64 (f64), shape (T, 1, H, W)
- Frames: dtype int16 (i16), shape (T, P, H, W)
- Time surface: dtype float64 (f64), shape (T, P, H, W)
- Denoise / Decimate: return structured event streams matching Python behavior

Rust fast paths preserve the shapes/dtypes above. Any non-bound case (e.g., certain 1D frame variants) intentionally falls back to Python.

### 5) Phase 1 acceptance

- All tests under `tonic-py/test/` pass with `TONIC_USE_RUST=1` for supported ops
- Fallback to Python is expected where bindings are intentionally not yet used (e.g., frame 1D case)
- The runner returns pytest’s exit code

### 6) Troubleshooting

If you see:
```
Rust extension module 'tonic_python' is not importable in this Python environment.
```
Follow these steps in the same Python environment used to run the tests:
```bash
python -m pip install maturin
maturin develop -m tonic-python/pyproject.toml
```
Then re-run:
```bash
python scripts/run_pytests_with_rust.py --pytest -q
```

To verify the extension is importable:
```bash
python -c "import tonic_python, sys; print(tonic_python.__file__); print(sys.version)"
```

### Paths at a glance

- Runner: [scripts/run_pytests_with_rust.py](scripts/run_pytests_with_rust.py)
- Python extension config: [tonic-python/pyproject.toml](tonic-python/pyproject.toml)
- Python package under test: ../tonic-py/
- Test suite: ../tonic-py/test/
