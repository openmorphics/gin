use crate::events::Event;

/// Refractory period kernel.
///
/// Keeps at most one event per pixel within the refractory window.
/// For each pixel (x,y), an event at time `t` is kept iff
///     t > last_kept_time[x,y] + delta
/// After keeping an event, last_kept_time[x,y] := t.
///
/// Semantics are aligned with tonic's RefractoryPeriod transform
/// (see docs/parity.md) and functional reference.
///
/// Safety & behavior:
/// - Input events must be timestamp-sorted (non-decreasing) for deterministic behavior.
/// - Out-of-bounds x/y are skipped.
/// - `sensor_h == 1` supports 1D sensors.
/// - Returns kept events in original order.
pub fn refractory_period(events: &[Event], sensor_w: usize, sensor_h: usize, delta: i64) -> Vec<Event> {
    if events.is_empty() || sensor_w == 0 || sensor_h == 0 {
        return Vec::new();
    }

    let w = sensor_w;
    let h = sensor_h;
    let hw = h.saturating_mul(w);

    // Initialize with i64::MIN: first event per pixel will always satisfy t > MIN + delta
    let mut last_kept = vec![i64::MIN; hw];

    let mut out = Vec::with_capacity(events.len());
    for ev in events {
        let x = ev.x as usize;
        let y = ev.y as usize;
        if x >= w || y >= h {
            continue;
        }
        let idx = y * w + x;
        let t = ev.t_ns;
        // Keep iff t > last + delta (avoid overflow via saturating add)
        if t > last_kept[idx].saturating_add(delta) {
            out.push(*ev);
            last_kept[idx] = t;
        }
    }

    out
}