use crate::events::Event;
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use rand_distr::Normal;

/// Time skew: apply a linear transform to timestamps: t' = t * coef + offset.
///
/// Notes:
/// - Input order is preserved; only timestamps are rewritten.
/// - Floating computation is truncated toward zero when cast to i64,
///   mirroring NumPy assignment of float into int64 fields.
pub fn time_skew(events: &[Event], coef: f64, offset: i64) -> Vec<Event> {
    if events.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(events.len());
    for ev in events {
        let v = (ev.t_ns as f64) * coef + (offset as f64);
        // Truncate toward zero (Rust as i64 cast truncates toward zero), matching NumPy semantics.
        let new_t = if v.is_finite() { v as i64 } else { ev.t_ns };
        out.push(Event {
            t_ns: new_t,
            ..*ev
        });
    }
    out
}

/// Time jitter: add Gaussian noise N(0, std^2) to each timestamp.
///
/// Parameters:
/// - std: standard deviation of the jitter (f64)
/// - clip_negative: if true, drop events with t' < 0
/// - sort_timestamps: if true, sort by t' ascending after jitter
///
/// Notes:
/// - When std == 0, timestamps are unchanged (except optional sort/clip behavior).
pub fn time_jitter(
    events: &[Event],
    std: f64,
    clip_negative: bool,
    sort_timestamps: bool,
) -> Vec<Event> {
    if events.is_empty() {
        return Vec::new();
    }
    if std <= 0.0 {
        // No jitter; only apply clip/sort if requested.
        let mut out: Vec<Event> = events.to_vec();
        if clip_negative {
            out.retain(|e| e.t_ns >= 0);
        }
        if sort_timestamps {
            out.sort_by_key(|e| e.t_ns);
        }
        return out;
    }

    let normal = Normal::new(0.0, std).unwrap_or_else(|_| Normal::new(0.0, 1e-9).unwrap());
    let mut rng = thread_rng();

    let mut out = Vec::with_capacity(events.len());
    for ev in events {
        let v = (ev.t_ns as f64) + normal.sample(&mut rng);
        let new_t = if v.is_finite() { v as i64 } else { ev.t_ns };
        if clip_negative && new_t < 0 {
            continue;
        }
        out.push(Event {
            t_ns: new_t,
            ..*ev
        });
    }

    if sort_timestamps {
        out.sort_by_key(|e| e.t_ns);
    }
    out
}

/// Uniform noise: append `n` noise events uniformly sampled over
/// x ∈ [0, W), y ∈ [0, H), p ∈ [0, P), and t ∈ [min_t, max_t].
///
/// The result is concatenated and then sorted by t ascending.
///
/// Notes:
/// - If input is empty, this returns exactly `n` noise events (with t in [0, 0]).
/// - Bounds safety: requires sensor_w>0 and sensor_h>0 to generate valid x/y.
pub fn uniform_noise(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    n: usize,
    p_channels: usize,
) -> Vec<Event> {
    let mut rng = thread_rng();

    let (min_t, max_t) = if events.is_empty() {
        (0i64, 0i64)
    } else {
        let first_t = events.first().unwrap().t_ns;
        let last_t = events.last().unwrap().t_ns;
        (first_t.min(last_t), first_t.max(last_t))
    };

    let mut out = Vec::with_capacity(events.len().saturating_add(n));
    out.extend_from_slice(events);

    if n > 0 && sensor_w > 0 && sensor_h > 0 && p_channels > 0 {
        let ux = Uniform::from(0..sensor_w);
        let uy = Uniform::from(0..sensor_h);
        let up = Uniform::from(0..p_channels);

        // t is sampled uniformly in the inclusive range [min_t, max_t].
        let ut = if min_t == max_t {
            None
        } else {
            Some(Uniform::from(min_t..=max_t))
        };

        for _ in 0..n {
            let x = ux.sample(&mut rng) as u16;
            let y = uy.sample(&mut rng) as u16;
            let p = up.sample(&mut rng) as i8;
            let t = if let Some(ref ut_) = ut {
                ut_.sample(&mut rng)
            } else {
                min_t
            };
            out.push(Event {
                t_ns: t,
                x,
                y,
                p,
                stream_id: 0,
            });
        }
    }

    // Sort final stream by timestamp to match NumPy reference behavior.
    out.sort_by_key(|e| e.t_ns);
    out
}

/// Time reverse: reflect timestamps over max(t) and reverse order.
///
/// Matches Python RandomTimeReversal (event path):
/// - Compute t' = max(t) - t for each event
/// - Output order is reversed relative to input
/// - x, y, p are unchanged
/// - Deterministic; no RNG; empty input -> empty output
pub fn time_reverse(events: &[Event]) -> Vec<Event> {
    if events.is_empty() {
        return Vec::new();
    }
    let t_max = events.iter().map(|e| e.t_ns).max().unwrap();
    let mut out = Vec::with_capacity(events.len());
    for ev in events.iter().rev() {
        out.push(Event {
            t_ns: t_max - ev.t_ns,
            ..*ev
        });
    }
    out
}