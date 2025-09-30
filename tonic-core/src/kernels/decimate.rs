use crate::events::Event;

/// Decimate kernel: keep every n-th event per pixel, preserving order.
///
/// Behavior:
/// - Maintain a per-pixel counter initialized to 0.
/// - For each event at (x, y): increment counter; when counter >= n, reset counter to 0 and keep the event; otherwise drop it.
/// - Supports 1D sensors via sensor_h == 1.
/// - Out-of-range x/y are skipped for safety.
///
/// Panics:
/// - Asserts that n > 0 (parity with Python implementation).
pub fn decimate(events: &[Event], sensor_w: usize, sensor_h: usize, n: usize) -> Vec<Event> {
    assert!(n > 0, "n has to be an integer greater than zero.");

    if events.is_empty() || sensor_w == 0 || sensor_h == 0 {
        return Vec::new();
    }

    let w = sensor_w;
    let h = sensor_h;
    let hw = h.saturating_mul(w);

    // Per-pixel counters
    let mut counters = vec![0usize; hw];
    let mut out = Vec::with_capacity(events.len() / n.max(1));

    for ev in events {
        let x = ev.x as usize;
        let y = ev.y as usize;
        if x >= w || y >= h {
            continue;
        }
        let idx = y * w + x;
        counters[idx] += 1;
        if counters[idx] >= n {
            counters[idx] = 0;
            out.push(*ev);
        }
    }

    out
}