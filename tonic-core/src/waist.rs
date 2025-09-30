//! Thin-waist interfaces for Tonic-rs.
//!
//! This module defines the minimal contracts we standardize on:
//! - EventSource: producers of time-sorted, columnar event chunks
//! - Transform: pure, batchable transforms applied to samples
//! - Eval: streaming evaluator producing deterministic metrics
//!
//! Notes
//! - Event tensors use a fixed schema: (t_ns: i64, x: u16, y: u16, p: i8, stream_id: u16)
//! - Inputs must be time-sorted (non-decreasing) within each stream_id
//! - Polarity convention for kernels: p ∈ {0, 1}; voxelization maps 0 → -1.0
//!
//! This module intentionally has no dependency on Arrow/DLPack; adapters live elsewhere.

use core::cmp::Ordering;
use std::collections::BTreeMap;

/// Columnar view over an event batch (no ownership).
///
/// Invariants:
/// - All columns have identical length N
/// - t_ns is non-decreasing globally (recommended: per-stream_id monotonic)
/// - p values are conventionally in {0, 1}
pub struct EventChunk<'a> {
    pub t_ns: &'a [i64],
    pub x: &'a [u16],
    pub y: &'a [u16],
    pub p: &'a [i8],
    pub stream_id: &'a [u16],
}

impl<'a> EventChunk<'a> {
    /// Number of events in this chunk.
    #[inline]
    pub fn len(&self) -> usize {
        self.t_ns.len()
    }

    /// Whether the chunk is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Validate column lengths and optional monotonicity of timestamps.
    ///
    /// Checks:
    /// - All columns share the same length
    /// - t_ns is non-decreasing (global) if `check_monotonic` is true
    pub fn validate(&self, check_monotonic: bool) -> Result<(), &'static str> {
        let n = self.t_ns.len();
        if self.x.len() != n
            || self.y.len() != n
            || self.p.len() != n
            || self.stream_id.len() != n
        {
            return Err("EventChunk columns must have identical lengths");
        }
        if check_monotonic && n > 1 {
            // Global non-decreasing check. Producers should still aim for per-stream monotonicity.
            for i in 1..n {
                if self.t_ns[i] < self.t_ns[i - 1] {
                    return Err("EventChunk.t_ns must be non-decreasing");
                }
            }
        }
        Ok(())
    }

    /// Returns whether t_ns is globally non-decreasing.
    #[inline]
    pub fn is_monotonic_non_decreasing(&self) -> bool {
        let n = self.t_ns.len();
        if n <= 1 {
            return true;
        }
        for i in 1..n {
            if self.t_ns[i].cmp(&self.t_ns[i - 1]) == Ordering::Less {
                return false;
            }
        }
        true
    }
}

/// Producers of columnar event batches.
///
/// Contract:
/// - Each returned chunk must satisfy EventChunk::validate(true)
/// - Successive chunks from the same source must not overlap in time
/// - Implementations should document shard provenance (min/max t_ns)
pub trait EventSource {
    fn next_chunk<'a>(&'a mut self) -> Option<EventChunk<'a>>;
}

/// Pure transform over a sample `S`.
///
/// Determinism requirements:
/// - `apply` must be referentially transparent given identical inputs and transform params.
pub trait Transform<S> {
    fn apply(&self, sample: &mut S);
}

/// Deterministic metrics keyed by name.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metrics(pub BTreeMap<String, f64>);

impl Metrics {
    #[inline]
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    #[inline]
    pub fn insert(&mut self, k: impl Into<String>, v: f64) {
        self.0.insert(k.into(), v);
    }

    #[inline]
    pub fn get(&self, k: &str) -> Option<f64> {
        self.0.get(k).copied()
    }
}

/// Streaming evaluator interface.
///
/// Typical usage:
/// - call update() for each (prediction, ground-truth) pair
/// - call finalize() once to obtain aggregate metrics
pub trait Eval<Pred, GT> {
    fn update(&mut self, y_hat: &Pred, y: &GT);
    fn finalize(&self) -> Metrics;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_chunk_validate_lengths_and_monotonicity() {
        let t = [1i64, 2, 2, 5];
        let x = [0u16, 1, 1, 2];
        let y = [0u16, 0, 1, 1];
        let p = [1i8, 0, 1, 1];
        let sid = [0u16, 0, 0, 0];

        let chunk = EventChunk {
            t_ns: &t,
            x: &x,
            y: &y,
            p: &p,
            stream_id: &sid,
        };
        assert!(chunk.validate(true).is_ok());
        assert!(chunk.is_monotonic_non_decreasing());

        let bad_t = [2i64, 1];
        let chunk_bad = EventChunk {
            t_ns: &bad_t,
            x: &x[..2],
            y: &y[..2],
            p: &p[..2],
            stream_id: &sid[..2],
        };
        assert!(chunk_bad.validate(true).is_err());
        assert!(!chunk_bad.is_monotonic_non_decreasing());
    }

    struct DummySrc<'a> {
        chunk: Option<EventChunk<'a>>,
    }

    impl<'a> EventSource for DummySrc<'a> {
        fn next_chunk<'b>(&'b mut self) -> Option<EventChunk<'b>> {
            // For demo purposes only: hand back None after first call
            self.chunk.take().map(|_c| {
                // This is a compile-only demo; in real code retain lifetimes appropriately.
                // We can't move out of borrowed data here; just return None.
                // Keeping the trait usable for real implementations.
                // To satisfy type checker in this dummy, immediately return None:
                // (We use a different approach below.)
                unreachable!("demo EventSource should not be used at runtime");
            })
        }
    }

    #[test]
    fn metrics_basic_ops() {
        let mut m = Metrics::new();
        m.insert("acc", 0.99);
        assert_eq!(m.get("acc"), Some(0.99));
        assert!(m.get("f1").is_none());
    }
}