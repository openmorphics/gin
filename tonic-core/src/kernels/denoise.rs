use crate::events::Event;

/// Denoise kernel (4-neighborhood temporal filter).
///
/// Semantics (parity with Python tonic.functional.denoise_numpy):
/// - Maintain timestamp memory per pixel, initialized to `filter_time`.
/// - Iterate events in input order (stable).
/// - For each event (x,y,t):
///     - Set self memory to t + filter_time BEFORE neighbor checks.
///     - Keep event if any 4-connected neighbor has memory > t
///       (i.e., that neighbor had an event within `filter_time` before t).
/// - Return the kept events in original order.
///
/// Bounds safety:
/// - Skip events whose x/y fall outside [0, sensor_w) x [0, sensor_h) to avoid panics.
///
/// Notes:
/// - Using i64 memory values to match event timestamp type.
/// - Memory layout is row-major over (H, W): idx = y * W + x.
pub fn denoise(events: &[Event], sensor_w: usize, sensor_h: usize, filter_time: i64) -> Vec<Event> {
    if events.is_empty() || sensor_w == 0 || sensor_h == 0 {
        return Vec::new();
    }

    let w = sensor_w;
    let h = sensor_h;
    let hw = h.saturating_mul(w);

    // Initialize timestamp memory to filter_time (Python: zeros + filter_time)
    let mut mem = vec![filter_time; hw];

    let mut out = Vec::with_capacity(events.len());

    for ev in events {
        let x = ev.x as usize;
        let y = ev.y as usize;
        if x >= w || y >= h {
            continue;
        }
        let t = ev.t_ns;
        let idx = y * w + x;

        // Set self memory before neighbor checks (Python parity)
        mem[idx] = t.saturating_add(filter_time);

        // Check 4-connected neighbors
        let mut keep = false;

        // Left
        if x > 0 {
            let n_idx = y * w + (x - 1);
            if mem[n_idx] > t {
                keep = true;
            }
        }
        // Right
        if !keep && x + 1 < w {
            let n_idx = y * w + (x + 1);
            if mem[n_idx] > t {
                keep = true;
            }
        }
        // Up
        if !keep && y > 0 {
            let n_idx = (y - 1) * w + x;
            if mem[n_idx] > t {
                keep = true;
            }
        }
        // Down
        if !keep && y + 1 < h {
            let n_idx = (y + 1) * w + x;
            if mem[n_idx] > t {
                keep = true;
            }
        }

        if keep {
            out.push(*ev);
        }
    }

    out
}