#!/usr/bin/env python3
"""
Conformance runner: compare Python reference (NumPy) vs Rust-accelerated outputs
for the supported kernels.

This script generates a deterministic synthetic event stream, runs a selected op
twice in child processes:
  - Baseline:  TONIC_USE_RUST=0  (Python-only)
  - Candidate: TONIC_USE_RUST=1  (Rust fast path via tonic_python)

It then compares results exactly (int) or within epsilon (float) and exits 0 on
success, non-zero on mismatch.

Examples:
  # Frames by time window
  python tonic-rs/scripts/conformance.py --op to_frame_time_window --n-events 5000 --w 64 --h 48 --time-window 3000

  # Voxel grid
  python tonic-rs/scripts/conformance.py --op voxel_grid --n-events 5000 --w 64 --h 48 --time-bins 10

  # Time surface
  python tonic-rs/scripts/conformance.py --op time_surface --n-events 5000 --w 64 --h 48 --dt 1000 --tau 1000

  # Denoise
  python tonic-rs/scripts/conformance.py --op denoise --n-events 5000 --w 64 --h 48 --filter-time 100

  # Decimate
  python tonic-rs/scripts/conformance.py --op decimate --n-events 5000 --w 64 --h 48 --n 5
"""

import argparse
import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


# ----------------------------
# Utilities for event handling
# ----------------------------

def make_struct_dtype() -> np.dtype:
    # Match tonic-py struct fields: x:int16, y:int16, t:int64, p:bool
    return np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.bool_)])


def gen_events(n: int, w: int, h: int, seed: int = 0xC0DEBEEF) -> np.ndarray:
    """Generate a deterministic structured event array with monotonically increasing timestamps."""
    rng = np.random.default_rng(seed)
    dtype = make_struct_dtype()
    ev = np.empty(n, dtype=dtype)

    # linearly spaced timestamps in [0, duration]
    duration = 1_000_000
    if n <= 1:
        ts = np.array([0], dtype=np.int64)
    else:
        ts = np.floor(np.linspace(0, duration, num=n)).astype(np.int64)
    ev["t"] = ts

    ev["x"] = rng.integers(low=0, high=max(1, w), size=n, dtype=np.int16)
    ev["y"] = rng.integers(low=0, high=max(1, h), size=n, dtype=np.int16)
    ev["p"] = rng.integers(low=0, high=2, size=n, dtype=np.int8).astype(bool)
    return ev


def save_events_npz(path: Path, events: np.ndarray, sensor_size: Tuple[int, int, int]) -> None:
    np.savez_compressed(path, x=events["x"], y=events["y"], t=events["t"], p=events["p"],
                        W=np.int64(sensor_size[0]), H=np.int64(sensor_size[1]), P=np.int64(sensor_size[2]))


def load_events_npz(path: Path) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    data = np.load(path, allow_pickle=False)
    dtype = make_struct_dtype()
    x = data["x"].astype(np.int16, copy=False)
    y = data["y"].astype(np.int16, copy=False)
    t = data["t"].astype(np.int64, copy=False)
    p = data["p"].astype(bool, copy=False)
    events = np.empty_like(t, dtype=dtype)
    events["x"] = x
    events["y"] = y
    events["t"] = t
    events["p"] = p
    W = int(data["W"])
    H = int(data["H"])
    P = int(data["P"])
    return events, (W, H, P)


def save_output_npz_dense(path: Path, arr: np.ndarray) -> None:
    np.savez_compressed(path, kind=np.string_("dense"), dtype=str(arr.dtype), shape=np.array(arr.shape, dtype=np.int64), arr=arr)


def save_output_npz_events(path: Path, events: np.ndarray) -> None:
    np.savez_compressed(path, kind=np.string_("events"),
                        x=events["x"].astype(np.uint16, copy=False),
                        y=events["y"].astype(np.uint16, copy=False),
                        t=events["t"].astype(np.int64, copy=False),
                        p=events["p"].astype(np.int8, copy=False))


def load_output_npz(path: Path) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=False)
    kind = str(d["kind"].astype(str))
    out: Dict[str, Any] = {"kind": kind}
    if kind == "dense":
        arr = d["arr"]
        out["arr"] = arr
    elif kind == "events":
        dtype = np.dtype([("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.int8)])
        n = d["t"].shape[0]
        ev = np.empty(n, dtype=dtype)
        ev["x"] = d["x"]
        ev["y"] = d["y"]
        ev["t"] = d["t"]
        ev["p"] = d["p"]
        out["events"] = ev
    else:
        raise RuntimeError(f"Unknown output kind: {kind}")
    return out


# ----------------------------
# Child runner
# ----------------------------

def child_run(op: str, events_path: Path, out_path: Path, args: argparse.Namespace) -> int:
    # Import tonic lazily in child
    import importlib
    import numpy as _np  # noqa: F401

    events, (W, H, P) = load_events_npz(events_path)
    # Sanity: ensure increasing timestamps
    if events.shape[0] > 1:
        idx = np.argsort(events["t"], kind="stable")
        events = events[idx]

    # Functional calls
    import tonic.functional as F

    if op == "to_frame_time_window":
        arr = F.to_frame_numpy(
            events=events,
            sensor_size=(W, H, P),
            time_window=int(args.time_window),
            event_count=None,
            n_time_bins=None,
            n_event_bins=None,
            overlap=float(args.overlap),
            include_incomplete=bool(args.include_incomplete),
            start_time=None,
            end_time=None,
        )
        save_output_npz_dense(out_path, arr.astype(np.int16, copy=False))
        return 0

    if op == "to_frame_event_count":
        arr = F.to_frame_numpy(
            events=events,
            sensor_size=(W, H, P),
            time_window=None,
            event_count=int(args.event_count),
            n_time_bins=None,
            n_event_bins=None,
            overlap=int(args.overlap),
            include_incomplete=bool(args.include_incomplete),
            start_time=None,
            end_time=None,
        )
        save_output_npz_dense(out_path, arr.astype(np.int16, copy=False))
        return 0

    if op == "to_frame_n_time_bins":
        arr = F.to_frame_numpy(
            events=events,
            sensor_size=(W, H, P),
            time_window=None,
            event_count=None,
            n_time_bins=int(args.time_bins),
            n_event_bins=None,
            overlap=float(args.overlap_frac),
            include_incomplete=False,
            start_time=None,
            end_time=None,
        )
        save_output_npz_dense(out_path, arr.astype(np.int16, copy=False))
        return 0

    if op == "to_frame_n_event_bins":
        arr = F.to_frame_numpy(
            events=events,
            sensor_size=(W, H, P),
            time_window=None,
            event_count=None,
            n_time_bins=None,
            n_event_bins=int(args.event_bins),
            overlap=float(args.overlap_frac),
            include_incomplete=False,
            start_time=None,
            end_time=None,
        )
        save_output_npz_dense(out_path, arr.astype(np.int16, copy=False))
        return 0

    if op == "voxel_grid":
        arr = F.to_voxel_grid_numpy(
            events=events,
            sensor_size=(W, H, P),
            n_time_bins=int(args.time_bins),
        )
        save_output_npz_dense(out_path, arr.astype(np.float64, copy=False))
        return 0

    if op == "time_surface":
        arr = F.to_timesurface_numpy(
            events=events,
            sensor_size=(W, H, P),
            dt=int(args.dt),
            tau=float(args.tau),
            overlap=int(args.overlap),
            include_incomplete=bool(args.include_incomplete),
        )
        save_output_npz_dense(out_path, arr.astype(np.float64, copy=False))
        return 0

    if op == "denoise":
        from tonic.functional import denoise_numpy
        out = denoise_numpy(events=events, filter_time=int(args.filter_time))
        save_output_npz_events(out_path, out.view(np.dtype([("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.int8)])))
        return 0

    if op == "decimate":
        from tonic.functional import decimate_numpy
        out = decimate_numpy(events=events, n=int(args.n))
        save_output_npz_events(out_path, out.view(np.dtype([("x", np.uint16), ("y", np.uint16), ("t", np.int64), ("p", np.int8)])))
        return 0

    print(f"Unknown op: {op}", file=sys.stderr)
    return 2


# ----------------------------
# Parent runner
# ----------------------------

def run_child(op: str, events_path: Path, out_path: Path, use_rust: bool) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "tonic-py") + os.pathsep + env.get("PYTHONPATH", "")
    env["TONIC_USE_RUST"] = "1" if use_rust else "0"
    # Narrow ops key for selectivity:
    if use_rust:
        # Map op to adapter key set
        if op.startswith("to_frame"):
            env["TONIC_RUST_OPS"] = "to_frame"
        elif op == "voxel_grid":
            env["TONIC_RUST_OPS"] = "voxel_grid"
        elif op == "time_surface":
            env["TONIC_RUST_OPS"] = "time_surface"
        elif op == "denoise":
            env["TONIC_RUST_OPS"] = "denoise"
        elif op == "decimate":
            env["TONIC_RUST_OPS"] = "decimate"

    # Marshal args we used in parent
    args = []
    # We propagate parameter CLI flags from parent argv (after a separator) to the child.
    # For simplicity, we serialize the argument namespace as JSON and pass via env.
    env["CONFORMANCE_ARGS_JSON"] = base64.b64encode(json.dumps(vars(PARENT_ARGS)).encode("utf-8")).decode("ascii")

    cmd = [sys.executable, __file__, "--mode", "child", "--op", op, "--events", str(events_path), "--out", str(out_path)]
    import subprocess
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def compare_outputs(baseline: Dict[str, Any], candidate: Dict[str, Any], eps: float) -> Tuple[bool, str]:
    if baseline["kind"] != candidate["kind"]:
        return False, f"Kind mismatch: {baseline['kind']} vs {candidate['kind']}"
    kind = baseline["kind"]
    if kind == "dense":
        a = baseline["arr"]
        b = candidate["arr"]
        if a.shape != b.shape:
            return False, f"Shape mismatch: {a.shape} vs {b.shape}"
        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
            diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
            if not np.isfinite(diff):
                return False, f"Non-finite diff encountered."
            if diff > eps:
                return False, f"Max abs diff {diff} > eps {eps}"
            return True, f"Float arrays match within eps={eps} (max diff={diff})."
        else:
            ok = np.array_equal(a, b)
            return (ok, "Exact int array match." if ok else "Integer arrays differ.")
    elif kind == "events":
        ea = baseline["events"]; eb = candidate["events"]
        if ea.shape != eb.shape:
            return False, f"Event count mismatch: {ea.shape} vs {eb.shape}"
        # Compare fields exactly
        ok = np.array_equal(ea["x"], eb["x"]) and np.array_equal(ea["y"], eb["y"]) \
             and np.array_equal(ea["t"], eb["t"]) and np.array_equal(ea["p"], eb["p"])
        return (ok, "Exact events match." if ok else "Event fields differ.")
    else:
        return False, f"Unhandled kind: {kind}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conformance runner for tonic kernels.")
    p.add_argument("--mode", choices=["parent", "child"], default="parent")
    p.add_argument("--op", required=True, help="One of: to_frame_time_window, to_frame_event_count, to_frame_n_time_bins, to_frame_n_event_bins, voxel_grid, time_surface, denoise, decimate")
    p.add_argument("--events", type=str, help="Path to input events npz (child mode)")
    p.add_argument("--out", type=str, help="Path to output npz (child mode)")

    # Common generation params (parent)
    p.add_argument("--n-events", type=int, default=5000)
    p.add_argument("--w", type=int, default=64)
    p.add_argument("--h", type=int, default=48)
    p.add_argument("--p-channels", type=int, default=2)
    p.add_argument("--seed", type=int, default=0xC0DEBEEF)

    # Frame/time-surface/voxel parameters
    p.add_argument("--time-window", type=int, default=3000)
    p.add_argument("--event-count", type=int, default=1000)
    p.add_argument("--include-incomplete", action="store_true", default=False)
    p.add_argument("--overlap", type=float, default=0.0)
    p.add_argument("--time-bins", type=int, default=5)
    p.add_argument("--event-bins", type=int, default=5)
    p.add_argument("--overlap-frac", type=float, default=0.0)
    p.add_argument("--dt", type=int, default=1000)
    p.add_argument("--tau", type=float, default=1000.0)
    p.add_argument("--filter-time", type=int, default=100)
    p.add_argument("--n", type=int, default=5)

    # Comparison
    p.add_argument("--eps", type=float, default=1e-12)

    return p.parse_args()


def child_main(args: argparse.Namespace) -> int:
    # Rehydrate args from parent (parameters) if present
    cfg_b64 = os.environ.get("CONFORMANCE_ARGS_JSON")
    if cfg_b64:
        cfg = json.loads(base64.b64decode(cfg_b64.encode("ascii")).decode("utf-8"))
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    return child_run(args.op, Path(args.events), Path(args.out), args)


def parent_main(args: argparse.Namespace) -> int:
    # Generate events
    events = gen_events(n=args.n_events, w=args.w, h=args.h, seed=args.seed)
    sensor_size = (args.w, args.h, args.p_channels)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ev_path = td / "events.npz"
        out_py = td / "out_py.npz"
        out_rs = td / "out_rs.npz"
        save_events_npz(ev_path, events, sensor_size)

        # Baseline (Python)
        rc_py = run_child(args.op, ev_path, out_py, use_rust=False)
        if rc_py != 0:
            print("Baseline (Python) child failed.", file=sys.stderr)
            return rc_py

        # Candidate (Rust)
        rc_rs = run_child(args.op, ev_path, out_rs, use_rust=True)
        if rc_rs != 0:
            print("Candidate (Rust) child failed.", file=sys.stderr)
            return rc_rs

        base = load_output_npz(out_py)
        cand = load_output_npz(out_rs)
        ok, msg = compare_outputs(base, cand, args.eps)
        print(msg)
        if ok:
            print("PASS")
            return 0
        else:
            print("FAIL", file=sys.stderr)
            return 3


if __name__ == "__main__":
    args = parse_args()
    global PARENT_ARGS
    PARENT_ARGS = args  # captured by run_child
    if args.mode == "child":
        sys.exit(child_main(args))
    else:
        sys.exit(parent_main(args))