Throughput & memory: zero-copy, SIMD, cache-friendly layouts.

Determinism: strict timestamp ordering + tie-break rules baked in.

Streaming: async I/O, backpressure, time-windowed batching.

Interoperability: stable FFI to Python, plus Arrow/Parquet for everyone else.

Future bridge to EventFlow’s SAL/EIR (units, timing semantics, conformance) .

I’ll split changes into data model, transforms/runtime, I/O & formats, interop, testing, and roadmap.

1) Data model upgrades
1.1 Strongly-typed time & units (no more footguns)

Define a Time newtype with units at the type level:

#[repr(transparent)]
pub struct Micros(pub i64);   // or Nanos(i64)


Carry {time_unit, time_origin} in a per-stream Meta. Enforce conversions at the API boundary.

Rationale: EventFlow’s IR is unit-checked and time is first-class; aligning now prevents downstream bugs.

1.2 Columnar first (SoA), with AoS views

Store events as columnar buffers for cache/SIMD:

pub struct DvsColumns<'a> {
    pub t: &'a [i64],      // ns or µs
    pub x: &'a [u16],
    pub y: &'a [u16],
    pub p: &'a [u8],       // 0/1
}


Provide zero-copy AoS views via bytemuck/repr(C) for users who want struct-like access.

Benefit: Faster kernels; still compatible with Python struct arrays when exported.

1.3 Canonical “event tensor” packing

Add a lossless packer to (t, index, value, meta):

pub struct Packed<'a> { pub t:&'a [i64], pub idx:&'a [u32], pub val:&'a [i8], pub meta: Meta }
pub fn pack_dvs(cols: &DvsColumns, width: u32) -> Packed;


Matches the SAL expectation of timestamped channel/magnitude streams for later EventFlow interop.

1.4 Deterministic ordering policy

Enforce stable sort by (t, y, x, p) at ingestion; expose ensure_strict_order().

Document the tie-break rule; make it opt-in hard enforcement for training reproducibility. (EventFlow depends on strict ordering for trace equivalence.)

2) Transforms & runtime
2.1 Iterator-style, zero-alloc transforms

Design transforms as composable iterators over columnar windows:

pub trait EventOp {
    fn apply<'a>(&self, in_: DvsColumns<'a>, meta:&Meta, out: &mut EventBuf) -> Result<()>;
    fn deterministic(&self) -> bool { true }
}


Provide a Compose that fuses passes and reuses scratch buffers (arena alloc).

2.2 SIMD & multi-threading

Hot ops (time cropping, polarity denoise, voxel binning) get:

SIMD via std::simd or packed_simd_2

Work-stealing splits by time stripes (Rayon)

Target hundreds of millions of events/sec on commodity CPUs.

2.3 Streaming windows + backpressure

First-class time windows (by_time(50.ms())) and count windows.

Async sources with tokio::Stream<Item = EventChunk>; bounded channels to avoid OOM.

Optional clock-sync hook (accept time beacons, expose drift estimates) to plug into a shared timebase later.

2.4 Deterministic modes

Exact-event mode: strict time order, pure ops → bitwise-stable results.

Fixed-step mode: Δt discretization for throughput; report quantization error.

Mirrors EventFlow’s two execution modes, easing future conformance.

3) I/O & on-disk formats
3.1 Arrow/Parquet as native store

Define Arrow schemas for DVS/audio/IMU; write Parquet with dictionary/meta:

self-describing, columnar, fast filters, cross-language.

Keep readers for AEDAT/h5, but normalize immediately to Arrow columns.

3.2 Memory-mapped datasets

mmap Parquet columns or a custom .evc (event column) container.

Zero-copy slices per window; OS page cache does the heavy lifting.

3.3 Dataset manifests

Emit a JSON sidecar (schema, units, epoch, resolution, checksums, license, version).

Ready for future EventFlow Hub ingestion.

4) Python & ecosystem interop
4.1 PyO3 bindings with zero-copy NumPy/Arrow

Expose views as NumPy arrays without copying (export the columnar buffers).

Optionally return PyArrow Tables → Pandas/Polars users are happy.

4.2 Drop-in Tonic API shim

A tonic_rs Python wheel exposing datasets.* and transforms.* with near-identical signatures.

Keeps existing notebooks working while gaining Rust speed.

4.3 Torch integration

Provide to_voxel_grid() returning contiguous ndarray ready for torch.from_numpy() in Python, or a direct DLPack/Tensor API on the Rust side for tch/PyTorch bindings.

5) Testing, QA, and conformance
5.1 Golden traces & ε-equivalence

For each transform, ship small golden inputs and check outputs bitwise (exact mode) or within ε (fixed-step).

Include trace replay (deterministic seeds), inspired by EventFlow’s conformance plan.

5.2 Profiles & capability tags

Tag transforms/datasets: BASE, REALTIME, LEARNING, LOWPOWER.

CI checks max latency/memory for REALTIME; document minimal CPU features.

6) Nice-to-have accelerators

GPU voxelization/time-surface via wgpu (portable) or CUDA feature-gated crates.

WASM build: run light transforms in the browser for demos/teaching.

Learned thresholds: tiny Rust-compiled model to adapt per-pixel C(x,y) in synthetic eventization.

7) API sketches (Rust)
Event columns + strict ordering
pub struct Meta {
    pub time_unit: TimeUnit,     // Ns | Us | Ms
    pub time_origin: TimeOrigin, // Unix | StreamStart
    pub width: u16,
    pub height: u16,
    pub channels: u8,
}

pub fn ensure_strict_order(mut t:&mut [i64], mut x:&mut [u16], mut y:&mut [u16], mut p:&mut [u8]) {
    // stable lexsort by (t, y, x, p); implement with index permutation to avoid moving large arrays repeatedly
}

Windowed iterator
pub struct ByTime<'a> { src: DvsColumns<'a>, win_us: i64, /* … */ }
impl<'a> Iterator for ByTime<'a> {
    type Item = DvsColumns<'a>;
    fn next(&mut self) -> Option<Self::Item> { /* slice indices without copying */ }
}

Transform trait with fuseable compose
pub trait EventOp { fn apply<'a>(&self, in_: DvsColumns<'a>, meta:&Meta, out:&mut EventBuf) -> Result<()>; }
pub struct Compose<T: EventOp>(Vec<T>);
impl<T: EventOp> EventOp for Compose<T> { /* run ops using a shared scratch */ }

8) Migration UX

Identical dataset names & documented parity table (“old Tonic → Tonic-RS”).

One-liner to get Arrow/Parquet:

import tonic_rs as tr
tr.convert("DVSGesture", out="dvsg.parquet", time_unit="us")


Determinism switch:

tr.set_mode("exact")  # or "fixed_step:1000ns"

9) Phased roadmap

Phase 0 (infra): Arrow schemas, Parquet writer/reader, core types (2–3 weeks).

Phase 1 (datasets): DVSGesture/N-MNIST/N-Caltech loaders + strict order + unit metadata.

Phase 2 (transforms): CropTime, Denoise, ToVoxelGrid (SIMD), TimeSurface; streaming windows.

Phase 3 (interop): PyO3, NumPy zero-copy, Torch DLPack, Python API shim.

Phase 4 (conformance): Golden traces, ε-equivalence, profile tags.

Phase 5 (extras): GPU voxelization, WASM demos, learned thresholding.