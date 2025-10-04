# Parity specification: Tonic Python ↔ Tonic Rust

This document defines behavioral and API parity between the Python implementation (“tonic”) and the Rust implementation (“tonic-rs”) for the kernels currently accelerated via the Python adapter.

Kernels in scope

- Frames (time-window, event-count, fixed time bins, fixed event bins)
- Voxel grid (time-interpolated event volumes)
- Time surface (recency-decayed surfaces)
- Denoise (4-neighborhood temporal filter)
- Decimate (keep every n-th event per pixel)

Primary references

- Python functional entrypoints:
  - [python.to_frame_numpy()](tonic-py/tonic/functional/to_frame.py:23)
  - [python.to_voxel_grid_numpy()](tonic-py/tonic/functional/to_voxel_grid.py:16)
  - [python.to_timesurface_numpy()](tonic-py/tonic/functional/to_timesurface.py:21)
  - [python.denoise_numpy()](tonic-py/tonic/functional/denoise.py:16)
  - [python.decimate_numpy()](tonic-py/tonic/functional/decimate.py:16)
- Rust kernels:
  - [rust.to_frame_time_window()](tonic-rs/tonic-core/src/kernels/frame.rs:92)
  - [rust.to_frame_event_count()](tonic-rs/tonic-core/src/kernels/frame.rs:147)
  - [rust.to_frame_n_time_bins()](tonic-rs/tonic-core/src/kernels/frame.rs:186)
  - [rust.to_frame_n_event_bins()](tonic-rs/tonic-core/src/kernels/frame.rs:226)
  - [rust.to_voxel_grid()](tonic-rs/tonic-core/src/kernels/voxel_grid.rs:12)
  - [rust.to_time_surface()](tonic-rs/tonic-core/src/kernels/time_surface.rs:25)
  - [rust.denoise()](tonic-rs/tonic-core/src/kernels/denoise.rs:20)
  - [rust.decimate()](tonic-rs/tonic-core/src/kernels/decimate.rs:13)
- Slicers (shared policy):
  - [rust.slice_by_time()](tonic-rs/tonic-core/src/slicers/slicers.rs:21)
  - [rust.slice_by_count()](tonic-rs/tonic-core/src/slicers/slicers.rs:83)
  - [rust.slice_by_time_bins()](tonic-rs/tonic-core/src/slicers/slicers.rs:121)
  - [rust.slice_by_event_bins()](tonic-rs/tonic-core/src/slicers/slicers.rs:156)
- Python↔Rust adapter:
  - [python._rust_backend](tonic-py/tonic/_rust_backend.py:1)
  - [rust PyO3 bindings](tonic-rs/tonic-python/src/lib.rs:1)

Event representation and dtype

- Python events use a structured ndarray with fields (x:int16, y:int16, t:int64, p:bool) defined in [python.events_struct](tonic-py/tonic/io.py:8).
- Rust kernels operate on [rust.Event](tonic-rs/tonic-core/src/events.rs:1) with fields t_ns:i64, x:u16, y:u16, p:i8, stream_id:u16.
- The Python adapter extracts contiguous 1D arrays, normalizes dtypes, and rebuilds structured arrays for filtered outputs where applicable [python._extract_fields](tonic-py/tonic/_rust_backend.py:76).

Timestamp ordering requirement

- All kernels assume non-decreasing timestamps. Slicers use binary search over ascending times ([rust slicers](tonic-rs/tonic-core/src/slicers/slicers.rs:1)).
- Recommended tie-break for strict determinism (to be enforced by a helper): stable lexsort by (t, y, x, p).

Polarity semantics

- Frames: polarity is a channel index p∈{0,1}. The adapter maps p>0→1 else 0 before calling Rust ([python p01 mapping](tonic-py/tonic/_rust_backend.py:142)). A single-polarity sensor (P=1) must not receive mixed polarities; otherwise error ([rust.check_single_polarity_rule](tonic-rs/tonic-core/src/kernels/frame.rs:15)).
- Voxel grid: contributions are ±1. Python converts 0→-1 in-place; Rust maps p==0→-1.0 internally ([rust.to_voxel_grid](tonic-rs/tonic-core/src/kernels/voxel_grid.rs:64)).
- Time surface, denoise, decimate: p is carried through; kernels ignore p except for channel indexing (time surface) or pass-through in filtered outputs.

Slicing policies (definitions)

1) Time-window slicing (frames, time surface)
   - Inputs: time_window (Δt), overlap (Δo), include_incomplete, optional start_time, end_time.
   - Stride = Δt - Δo must be > 0.
   - Window i: [start + i·stride, start + i·stride + Δt), with start/end defaults to first/last event time.
   - Number of slices:
     - include_incomplete=False: floor((duration - Δt)/stride) + 1, minimum 1
     - include_incomplete=True:  ceil((duration - Δt)/stride) + 1, minimum 1
   - Reference: [rust.slice_by_time()](tonic-rs/tonic-core/src/slicers/slicers.rs:21), tests in [rust.frame tests](tonic-rs/tonic-core/tests/frame.rs:76) and [rust.time_surface tests](tonic-rs/tonic-core/tests/time_surface.rs:116).

2) Event-count slicing (frames)
   - Inputs: event_count (N), overlap (o), include_incomplete.
   - Stride = N - o must be > 0.
   - Number of slices:
     - include_incomplete=False: floor((n_events - N)/stride) + 1
     - include_incomplete=True:  ceil((n_events - N)/stride) + 1
   - Reference: [rust.slice_by_count()](tonic-rs/tonic-core/src/slicers/slicers.rs:83), tests in [rust.frame tests](tonic-rs/tonic-core/tests/frame.rs:37).

3) Fixed number of time bins (frames)
   - Inputs: n_time_bins (T), overlap_frac ∈ [0,1).
   - Base = duration / T; Δt = floor(base·(1+overlap_frac)); stride = floor(Δt·(1-overlap_frac)).
   - Exactly T slices starting at t_first with given stride ([rust.slice_by_time_bins()](tonic-rs/tonic-core/src/slicers/slicers.rs:121)).

4) Fixed number of event bins (frames)
   - Inputs: n_event_bins (T), overlap_frac ∈ [0,1).
   - Base = n_events // T; spike_count = floor(base·(1+overlap_frac)); stride = floor(spike_count·(1-overlap_frac)).
   - Exactly T slices over indices ([rust.slice_by_event_bins()](tonic-rs/tonic-core/src/slicers/slicers.rs:156)).

Kernel semantics summary

- Frames
  - Accumulate counts per (t-slice, p, y, x) in i32, cast to i16 on return. Out-of-bounds x/y skipped. Single-polarity rule enforced. See [rust.frame kernel](tonic-rs/tonic-core/src/kernels/frame.rs:34).
- Voxel grid
  - Normalize time to [0, T]; bilinear accumulate into left/right adjacent bins; polarity ±1. Layout (T,1,H,W). See [rust.voxel_grid](tonic-rs/tonic-core/src/kernels/voxel_grid.rs:12).
- Time surface
  - Maintain last event time per (p,y,x); surface at slice i is exp(-(t_i - last_t)/τ), else 0. See [rust.time_surface](tonic-rs/tonic-core/src/kernels/time_surface.rs:25).
- Denoise
  - For each event, set self memory to t+filter_time; keep if any 4-neighbor has memory>t. See [rust.denoise](tonic-rs/tonic-core/src/kernels/denoise.rs:20).
- Decimate
  - Per-pixel counter; keep when counter≥n then reset; n>0. See [rust.decimate](tonic-rs/tonic-core/src/kernels/decimate.rs:13).

Empty-input behavior

- Frames (time_window, event_count): return zero slices (T=0, empty buffer) in Rust; Python user transform may return zeros with known sensor_size ([python.ToFrame.__call__](tonic-py/tonic/transforms.py:905)).
- Frames (fixed bins): always return exactly T zeros-filled slices in Rust ([rust.frame tests](tonic-rs/tonic-core/tests/frame.rs:275)).
- Voxel grid: returns zeros of shape (T,1,H,W) when events empty or degenerate window ([rust.voxel_grid](tonic-rs/tonic-core/src/kernels/voxel_grid.rs:21)).
- Time surface: zero slices when no windows (empty/degenerate) ([rust.time_surface](tonic-rs/tonic-core/src/kernels/time_surface.rs:37)).
- Denoise/Decimate: return empty filtered streams on empty inputs.

Error handling

- Python references raise ValueError/AssertionError for invalid parameters (e.g., multiple frame modes set, invalid overlaps).
- Rust kernels return Result<T, String>; PyO3 maps to PyValueError ([rust.bindings](tonic-rs/tonic-python/src/lib.rs:100)).
- Decimate asserts n>0 in Rust; binding pre-validates ([rust.decimate_py](tonic-rs/tonic-python/src/lib.rs:294)).

Layout and dtype of outputs

- Frames: int16 array (T,P,H,W), row-major (C-order).
- Voxel grid: float64 array (T,1,H,W), row-major.
- Time surface: float64 array (T,P,H,W), row-major.
- Denoise/Decimate: structured arrays with original dtype (x:u16, y:u16, t:i64, p:i8) via reconstruction in adapter ([python._reconstruct_structured](tonic-py/tonic/_rust_backend.py:212)).

Conformance methodology

- Compute reference with Rust disabled: set TONIC_USE_RUST=0, call Python functional op (e.g., [python.to_frame_numpy()](tonic-py/tonic/functional/to_frame.py:23)).
- Compute candidate with Rust enabled: set TONIC_USE_RUST=1 (and TONIC_RUST_OPS for the specific op), call the same functional op.
- Compare:
  - Int outputs: exact equality (bitwise).
  - Float outputs (voxel grid, time surface): |a-b| ≤ ε (default 1e-12 for tests).
- The conformance runner lives under scripts/ and is invoked similarly to [scripts/run_pytests_with_rust.py](tonic-rs/scripts/run_pytests_with_rust.py:1).

Known intentional differences (documented)

- Frames empty-input behavior: Rust kernels return T=0 for time_window/event_count; Python ToFrame may return zeros with known sensor_size. This is preserved for backward-compat and documented in user API ([python.ToFrame.__call__](tonic-py/tonic/transforms.py:905)).
- Polarity normalization happens at adapter boundary for frames to ensure p∈{0,1} ([python._rust_backend](tonic-py/tonic/_rust_backend.py:142)).

Future enhancements tracked

- Strict ordering helper: ensure_strict_order(t,x,y,p) and tie-break policy implementation near [rust.events](tonic-rs/tonic-core/src/events.rs:1) with tests.
- Arrow/Parquet event schemas and memory-mapped datasets per [tonic-rs/readme.md](tonic-rs/readme.md:98).

Contact and contributions

- Please open issues/PRs against tonic-rs with failing minimal examples and reference to the specific Python function and parameters.

## Drop-pixel and hot-pixel detection parity

- Frequency definition:
  - Frequency is defined as count(x,y) divided by the recording duration, where duration = max(t) - min(t). Units follow the event timestamp units used throughout the kernels.
- Deterministic outputs:
  - Hot-pixel coordinate outputs are deterministically ordered lexicographically by (y, then x).
- Empty and degenerate inputs:
  - Empty inputs yield an empty set of hot pixels.
  - If duration ≤ 0 (all timestamps equal), the hot-pixel set is empty.
  - Dropping with an empty coordinate list returns the original events unmodified; event order is preserved.

References:
- Python reference: [tonic-py/tonic/functional/drop_pixel.py](tonic-py/tonic/functional/drop_pixel.py)
- Rust kernels: [tonic-rs/tonic-core/src/kernels/drop_ops.rs](tonic-rs/tonic-core/src/kernels/drop_ops.rs)

## CenterCrop parity

- Offsets (floor division): ox = (sensor_w - target_w) // 2, oy = (sensor_h - target_h) // 2.
- Clamp target to sensor: target_w = min(target_w, sensor_w), target_h = min(target_h, sensor_h). Empty if any dimension is zero.
- Keep region: x ∈ [ox, ox+target_w), y ∈ [oy, oy+target_h). Rebase coordinates x' = x - ox, y' = y - oy.
- Preserve original order; skip out-of-bounds for safety.
- 1D sensors: sensor_h == 1 is supported (oy == 0).

References:
- Python functional: [python.crop_numpy()](tonic-py/tonic/functional/crop.py:4), transform [python.CenterCrop](tonic-py/tonic/transforms.py:44)
- Rust kernel: [rust.center_crop()](tonic-rs/tonic-core/src/kernels/crop.rs:1)
- PyO3 binding: [rust.center_crop_py()](tonic-rs/tonic-python/src/lib.rs:1)
- Python adapter: [python._rust_backend.center_crop()](tonic-py/tonic/_rust_backend.py:1)

## Flip parity

- Left-right (LR) flip:
  - Formula: x' = (W - 1) - x; y, t, p unchanged.
  - Events with x ∈ [0, W) and y ∈ [0, H) are transformed; out-of-bounds are skipped safely.
- Up-down (UD) flip:
  - Formula: y' = (H - 1) - y; x, t, p unchanged.
  - Events with x ∈ [0, W) and y ∈ [0, H) are transformed; out-of-bounds are skipped safely.
- Polarity flip:
  - Dual-encoding support:
    - If any event has p == 0 (0/1 encoding): p' = 1 - p for p ∈ {0,1}.
    - Else if any event has p == -1 (-1/1 encoding): p' = -p for p ∈ {-1,1}.
    - Else: p unchanged.
  - x, y, t remain unchanged.
- Determinism and order:
  - All flips are deterministic and preserve event order.
- Degenerate/edge cases:
  - Empty inputs return empty outputs without error.
  - 1D sensors are supported (e.g., H == 1 makes UD a no-op as y == 0 ⇒ y' == 0).
  - For valid in-bounds inputs, output length equals input length.

References:
- Rust kernels: [rust.flip_lr() / flip_ud() / flip_polarity()](tonic-rs/tonic-core/src/kernels/flip.rs:1)
- PyO3 bindings: [rust.flip_*_py()](tonic-rs/tonic-python/src/lib.rs:1)
- Python adapter routes: [python._rust_backend.flip_*](tonic-py/tonic/_rust_backend.py:1)
- Python transforms (fast-path wired): [python.RandomFlip*](tonic-py/tonic/transforms.py:488)

## RandomCrop parity

- Randomness and offset selection happen in Python. The transform samples integer offsets:
  - x0 = int(rand() · (W - target_w))
  - y0 = int(rand() · (H - target_h))
- Rust provides a deterministic primitive [crop::crop_box()](tonic-rs/tonic-core/src/kernels/crop.rs:1). The Python adapter routes to it when enabled; otherwise it falls back to the Python implementation.

Semantics of crop_box:

- Early returns:
  - Return empty if any of sensor_w, sensor_h, target_w, target_h is zero.
  - Return empty if x0 ≥ sensor_w or y0 ≥ sensor_h (origin outside sensor).
- Clamping:
  - Clamp to sensor edges: tw = min(target_w, sensor_w - x0), th = min(target_h, sensor_h - y0). Return empty if clamped tw or th becomes zero.
- Keep region and rebasing:
  - Keep events with x ∈ [x0, x0 + tw) and y ∈ [y0, y0 + th).
  - Rebase coordinates x' = x - x0, y' = y - y0.
- Safety and determinism:
  - Skip out-of-bounds events safely; preserve input order.
- 1D sensors:
  - sensor_h == 1 is supported; y0 must be 0. All semantics above still apply.

References:

- Python functional reference (random selection): [python.crop_numpy()](tonic-py/tonic/functional/crop.py:4)
- Rust kernel: [rust.crop::crop_box()](tonic-rs/tonic-core/src/kernels/crop.rs:1)
- PyO3 binding: [rust.crop_box_py()](tonic-rs/tonic-python/src/lib.rs:707)
- Python adapter route: [python._rust_backend.crop_box()](tonic-py/tonic/_rust_backend.py:507)
- Python transform (fast-path wired): [python.RandomCrop](tonic-py/tonic/transforms.py:424)

## RandomTimeReversal parity

- Scope and routing:
  - The Python transform [RandomTimeReversal.__call__()](tonic-py/tonic/transforms.py:635) implements two data paths:
    - Raster-like arrays shaped [t, p, h, w] or [t, p, x]: reversed purely in Python via `events[::-1, ::-1, ...]`. No Rust path is used for rasters.
    - Event streams (structured ndarray with fields x,y,t,p): eligible for Rust fast path through the adapter.
  - The Rust kernel lives in [tonic-rs/tonic-core/src/kernels/time_ops.rs](tonic-rs/tonic-core/src/kernels/time_ops.rs) as [time_ops.time_reverse()](tonic-rs/tonic-core/src/kernels/time_ops.rs:1). It is exported to Python via [time_reverse_py()](tonic-rs/tonic-python/src/lib.rs:1) and routed by [_rust_backend.time_reverse()](tonic-py/tonic/_rust_backend.py:1) when the op key "time_reverse" is enabled.

- Exact semantics (event stream path):
  - Compute reflection over the max timestamp:
    - Let t_max = max(events["t"])
    - For each event i: t'_i = t_max - t_i
  - Then reverse the sequence order (equivalent to NumPy `events[::-1]`).
  - x, y, p are unchanged by the kernel. The Python transform optionally flips polarities when `flip_polarities=True`:
    - In fast path, polarity flip happens in Python immediately after calling the Rust kernel so the overall result matches the reference behavior exactly.
  - Deterministic: no randomness in the kernel; input length equals output length; no drops or insertions.

- Ordering and ties:
  - The kernel does not perform sorting; it reflects timestamps and then reverses event order.
  - If the input timestamps are non-decreasing, then the output timestamps are also non-decreasing after reflection+reverse.
  - Equal timestamps remain equal; tie-break determinism follows the reverse of input index order.

- Empty input:
  - Empty input returns empty output.

- Dtypes and reconstruction:
  - Kernel carries x,y as u16, t as i64, p as i8 internally.
  - The Python adapter reconstructs a structured array of the original dtype and layout when fast path is used.

- Tests:
  - Parity is validated by [tonic-rs/tonic-core/tests/time_reverse.rs](tonic-rs/tonic-core/tests/time_reverse.rs:1) with:
    - Reflection + reverse mapping over a sample stream
    - Empty-input no-op
    - Stability of x,y,p fields
    - Edge case with equal timestamps (t_min == t_max) producing zeros after reflection with reversed order

## EventDownsampling parity

Scope and routing

- Fast-path key: "event_downsample" routed by [tonic._rust_backend](tonic-py/tonic/_rust_backend.py:1).
- Python functional entrypoints:
  - [integrator_downsample()](tonic-py/tonic/functional/event_downsampling.py:99)
  - [differentiator_downsample()](tonic-py/tonic/functional/event_downsampling.py:18)
- Transform wiring:
  - [EventDownsampling.__call__](tonic-py/tonic/transforms.py:386) attempts Rust fast-path first, then falls back to NumPy implementation.
- Rust core kernel:
  - [downsample.rs](tonic-rs/tonic-core/src/kernels/downsample.rs:1) provides:
    - [event_integrator_downsample()](tonic-rs/tonic-core/src/kernels/downsample.rs:47)
    - [event_differentiator_downsample()](tonic-rs/tonic-core/src/kernels/downsample.rs:125)
    - [event_downsample_integrator()](tonic-rs/tonic-core/src/kernels/downsample.rs:212) unified entry toggled by `differentiate`
- PyO3 binding and adapter:
  - [event_downsample_py()](tonic-rs/tonic-python/src/lib.rs:869) is exported to Python as tonic_python.event_downsample
  - Adapter router [_rust_backend.event_downsample(...)](tonic-py/tonic/_rust_backend.py:330) invokes the binding and reconstructs a structured ndarray

Semantics

- Spatial neighborhoods:
  - A spatial downsample factor f is derived in Python as:
    - f = W // target_w = H // target_h, requiring W % target_w == 0 and H % target_h == 0 and (W // target_w == H // target_h).
  - Reduced coordinates are computed by integer division: x' = floor(x/f), y' = floor(y/f).
  - Out-of-bounds input events (x ≥ W or y ≥ H) are ignored.
- Polarity contribution:
  - Signed unit contribution per input event in a reduced cell: +1 if p > 0, else -1 (0 is treated as negative).
  - Output polarity channel is encoded as p = 1 for positive spikes and p = 0 for negative spikes.
- Noise threshold policy:
  - threshold ≤ 0: pass-through mode — each in-bounds input event produces one output event mapped to its reduced cell with the corresponding sign.
  - threshold > 0:
    - Integrator mode (differentiate=False): per reduced cell accumulator integrates ±1; when |accum| ≥ threshold, emit a single spike at the current input event timestamp with p=1 for positive or p=0 for negative, then reset that cell accumulator to 0.
    - Differentiator mode (differentiate=True): maintain a per-cell thresholded state in {-1,0,+1} where new_state=+1 if accum ≥ threshold, -1 if -accum ≥ threshold, else 0. Emit an event only for rising-edge transitions 0→(+1) or 0→(-1), then reset the cell accumulator to 0. No events are emitted for (+1)→0 or (-1)→0 transitions.
- Determinism and ordering:
  - The kernel preserves input order deterministically and emits at most one output event per input event. No randomness is used.

Relationship to Python reference

- The Python integrator_downsample() computes time-sliced histograms using [to_frame_numpy()](tonic-py/tonic/functional/to_frame.py:23) with `dt` and emits spikes at slice boundaries after ON/OFF differencing. The Rust kernel emits spikes at the exact timestamp of the threshold-crossing input event while preserving equivalent integrate-and-fire and differentiator edge semantics over spatial neighborhoods.
- The Python differentiator_downsample() post-processes integrator outputs across `differentiator_time_bins` to detect inter-frame edges. The Rust differentiator kernel implements the same edge-selective behavior via per-cell state and accumulator discharge.
- Routing constraints in adapter/transform ensure that the Rust fast-path is only used when a consistent spatial factor f can be derived (equal factors along X and Y), otherwise it cleanly falls back to the Python NumPy implementation.

Edge cases

- Empty input: returns an empty event set.
- Out-of-bounds events: safely ignored.
- Degenerate sensors (W=0 or H=0): treated as empty.
- threshold ≤ 0: pass-through at reduced coordinates.

Dtypes and reconstruction

- Kernel uses x,y: u16, t: i64, p: i8 internally.
- The adapter reconstructs structured outputs to the original dtype layout when fast-path is taken.

Tests

- Rust unit tests: [tests/downsample.rs](tonic-rs/tonic-core/tests/downsample.rs:1) verify:
  - empty-input behavior
  - neighborhood mapping correctness
  - integrator accumulation and threshold firing
  - differentiator edge-emission behavior (+ and -)
  - noise threshold pass-through at 0
  - deterministic timestamp ordering
  - out-of-bounds handling
- Acceptance: `cargo test -p tonic-core` validates the kernel suite; Python transform behavior continues to match NumPy reference when routing falls back for unsupported configurations.

API surface

- Adapter key: "event_downsample" is added to the enabled-op set [_OPS](tonic-py/tonic/_rust_backend.py:22).
- Binding exported in module init: [tonic_python(...)](tonic-rs/tonic-python/src/lib.rs:910) registers [event_downsample_py()](tonic-rs/tonic-python/src/lib.rs:869).

Parameter ranges and dtype requirements

- sensor_size must be a 2-tuple (W:int, H:int) with W,H ≥ 0.
- downsample_factor (u8) is required by the kernel; the adapter derives it from sensor_size and target_size and only uses Rust when both axes share the same positive factor.
- noise_threshold is an integer; threshold ≤ 0 triggers pass-through at reduced resolution.

## ToAveragedTimesurface (HATS) parity

Scope and routing

- Functional reference: [to_averaged_timesurface_numpy()](tonic-py/tonic/functional/to_averaged_timesurface.py:55)
- Transform wiring: [ToAveragedTimesurface.__call__](tonic-py/tonic/transforms.py:925)
- Rust core kernel: [to_averaged_time_surface()](tonic-rs/tonic-core/src/kernels/averaged_time_surface.rs:48)
- PyO3 binding: [averaged_time_surface_py()](tonic-rs/tonic-python/src/lib.rs:245)
- Python adapter route: [_rust_backend.averaged_time_surface](tonic-py/tonic/_rust_backend.py:232)
- Module export: [kernels::averaged_time_surface](tonic-rs/tonic-core/src/kernels/mod.rs:4)

Semantics

- HATS histogram per (cell, polarity) averaged over events in that cell:
  - The sensor is partitioned into square cells of size cell_size.
  - For each cell c and polarity p independently, iterate events in input order (no sorting).
  - For each current event i in (c,p), consider all previous events j in the same (c,p) such that:
    - |x_j - x_i| ≤ ρ and |y_j - y_i| ≤ ρ where ρ = surface_size // 2
    - t_j ≥ max(0, t_i - time_window)
  - Place contributions at offset (dy = y_j - y_i, dx = x_j - x_i) in the local SxS surface:
    - decay="exp": exp(-(t_i - t_j)/τ)
    - decay="lin": -(t_i - t_j)/(3·τ) + 1
  - Add +1 at the center (ρ,ρ) for the current event i.
  - Accumulate these per-event surfaces over all events in (c,p), then divide element-wise by max(1, N_cp) where N_cp is the number of events in (c,p).
- Determinism:
  - The input event order defines “previous” events; no internal sorting is performed.
  - Out-of-bounds (x ≥ W or y ≥ H) are ignored safely.
  - Polarity mapping matches Python reference:
    - If P==1: all events map to channel 0.
    - If P&gt;1: negative p maps to channel 0; p ≥ P is skipped.

Parameters and dtype

- Inputs:
  - events fields: x:u16, y:u16, t:i64, p:i8 (structured ndarray in Python; [Event](tonic-rs/tonic-core/src/events.rs:1) in Rust)
  - sensor_size = (W, H, P)
  - cell_size:int &gt; 0
  - surface_size:int odd, surface_size ≤ cell_size
  - time_window: float (same units as t)
  - tau: float (same units as t)
  - decay: "exp" or "lin"
- Output: float32 ndarray with shape (C, P, S, S) where:
  - C = ceil(W / cell_size) * ceil(H / cell_size)
  - S = surface_size

Edge cases and error handling

- Empty input: returns zeros of shape (C, P, S, S).
- Degenerate sensor (W==0 or H==0 or P==0): returns empty buffer (C==0 or P==0 implies size 0).
- Validation errors:
  - cell_size == 0 → error
  - surface_size == 0 or even → error
  - surface_size &gt; cell_size → error
  - Unsupported decay string → PyValueError in binding.
- Tau edge cases:
  - τ is treated as a float; τ==0 follows IEEE behavior (may yield inf/NaN). Matches Python numpy semantics.

Adapter routing and fast path

- Fast-path key: "averaged_time_surface" (honored by TONIC_USE_RUST and TONIC_RUST_OPS).
- The transform attempts Rust fast path first; on failure, it silently falls back to NumPy:
  - Transform: [ToAveragedTimesurface.__call__](tonic-py/tonic/transforms.py:925)
  - Functional: [to_averaged_timesurface_numpy()](tonic-py/tonic/functional/to_averaged_timesurface.py:55) includes:
    - Fast-path attempt via [_rust_backend.averaged_time_surface](tonic-py/tonic/_rust_backend.py:232)
    - Fall back to the NumPy implementation when fast-path is disabled or unavailable.

Layout and shape

- Rust kernel returns a contiguous row-major buffer (C,P,S,S) that maps 1:1 to NumPy C-order.
- Python binding returns a float32 ndarray (C,P,S,S).

Tests

- Rust unit tests: [tests/averaged_time_surface.rs](tonic-rs/tonic-core/tests/averaged_time_surface.rs:1) validate:
  - Output shape (C,P,S,S)
  - Empty input returns zeros
  - Single-event center contribution averaged to 1
  - Temporal window inclusion/exclusion by time_window
  - Tau decay influence on averages (monotonic vs τ)
  - Deterministic outputs for identical inputs
