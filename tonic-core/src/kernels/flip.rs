// Deterministic flip kernels: LR (left-right), UD (up-down), and polarity.
//
// Coordinates policy:
// - Events with x in [0, W) and y in [0, H) are transformed; others are skipped safely.
// - Order is preserved for all kept events.
//
// Polarity policy:
// - Supports both common encodings with deterministic mapping:
//     • If the input contains any -1 and no 0 → treat as {-1, 1} encoding and map p' = -p for p ∈ {-1, 1}.
//     • Otherwise (default) treat as {0, 1} encoding and map p' = 1 - p for p ∈ {0, 1}.
//     • All other values are left unchanged.
// - This strategy mirrors Python's bool inversion semantics for typical {0,1} datasets,
//   while correctly handling pure {-1,1} datasets.
//
// Invariants for valid inputs:
// - LR/UD: coordinates remain within [0..W) / [0..H); order preserved.
// - Polarity: x,y,t unchanged; order preserved; output length == input length.
// - 1D sensors supported (e.g., H == 1 makes UD a no-op).
//
// Degenerate cases:
// - If sensor_w == 0 or sensor_h == 0, LR/UD return empty.
//
// Notes:
// - These functions are deterministic and contain no randomness.
use crate::Event;

#[inline]
pub fn flip_lr(events: &[Event], sensor_w: usize, sensor_h: usize) -> Vec<Event> {
    if sensor_w == 0 || sensor_h == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(events.len());
    for &ev in events {
        let x_u = ev.x as usize;
        let y_u = ev.y as usize;
        if x_u < sensor_w && y_u < sensor_h {
            let new_x = (sensor_w - 1 - x_u) as u16;
            out.push(Event {
                t_ns: ev.t_ns,
                x: new_x,
                y: ev.y,
                p: ev.p,
                stream_id: ev.stream_id,
            });
        } else {
            // Skip out-of-bounds safely
        }
    }
    out
}

#[inline]
pub fn flip_ud(events: &[Event], sensor_w: usize, sensor_h: usize) -> Vec<Event> {
    if sensor_w == 0 || sensor_h == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(events.len());
    for &ev in events {
        let x_u = ev.x as usize;
        let y_u = ev.y as usize;
        if x_u < sensor_w && y_u < sensor_h {
            let new_y = (sensor_h - 1 - y_u) as u16;
            out.push(Event {
                t_ns: ev.t_ns,
                x: ev.x,
                y: new_y,
                p: ev.p,
                stream_id: ev.stream_id,
            });
        } else {
            // Skip out-of-bounds safely
        }
    }
    out
}

#[inline]
pub fn flip_polarity(events: &[Event]) -> Vec<Event> {
    // Decide encoding mode once per call to avoid ambiguity for p == 1 across encodings.
    let has_zero = events.iter().any(|e| e.p == 0);
    let has_minus_one = events.iter().any(|e| e.p == -1);

    enum Mode {
        ZeroOne,     // {0,1} -> p' = 1 - p
        NegOneOne,   // {-1,1} -> p' = -p
        Passthrough, // leave unchanged
    }

    let mode = if has_minus_one && !has_zero {
        Mode::NegOneOne
    } else if has_zero {
        Mode::ZeroOne
    } else {
        // If neither 0 nor -1 are present, default to passthrough.
        Mode::Passthrough
    };

    let mut out = Vec::with_capacity(events.len());
    match mode {
        Mode::ZeroOne => {
            for &ev in events {
                let new_p = match ev.p {
                    0 => 1,
                    1 => 0,
                    other => other,
                };
                out.push(Event {
                    t_ns: ev.t_ns,
                    x: ev.x,
                    y: ev.y,
                    p: new_p,
                    stream_id: ev.stream_id,
                });
            }
        }
        Mode::NegOneOne => {
            for &ev in events {
                let new_p = match ev.p {
                    -1 => 1,
                    1 => -1,
                    other => other,
                };
                out.push(Event {
                    t_ns: ev.t_ns,
                    x: ev.x,
                    y: ev.y,
                    p: new_p,
                    stream_id: ev.stream_id,
                });
            }
        }
        Mode::Passthrough => {
            out.extend_from_slice(events);
        }
    }
    out
}