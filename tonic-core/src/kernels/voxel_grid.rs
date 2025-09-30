use crate::events::Event;

/// Build a voxel grid with bilinear interpolation in the time domain from a set of events.
///
/// Semantics aligned with the Python reference (to_voxel_grid_numpy) including:
/// - Normalization of timestamps to [0, n_time_bins]
/// - Bilinear accumulation into adjacent time bins (left/right)
/// - Polarity mapping: 0 -> -1.0, otherwise p as f64
/// - Layout: (T, 1, H, W) flattened in row-major (C-order)
///
/// Returns a Vec<f64> of length n_time_bins * 1 * sensor_h * sensor_w.
pub fn to_voxel_grid(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    n_time_bins: usize,
) -> Vec<f64> {
    let len = n_time_bins.saturating_mul(sensor_h).saturating_mul(sensor_w);
    let mut grid = vec![0.0f64; len];

    if n_time_bins == 0 || sensor_w == 0 || sensor_h == 0 {
        return grid;
    }
    if events.is_empty() {
        return grid;
    }

    let t0 = events.first().unwrap().t_ns;
    let t_last = events.last().unwrap().t_ns;
    let denom_i64 = t_last - t0;
    if denom_i64 <= 0 {
        // if t_last == t0 (or reversed), degenerate window - return zeros as per spec
        return grid;
    }
    let denom = denom_i64 as f64;
    let hw = sensor_h * sensor_w;

    for ev in events {
        let x = ev.x as usize;
        let y = ev.y as usize;

        debug_assert!(
            x < sensor_w && y < sensor_h,
            "Event (x={}, y={}) out of bounds (W={}, H={})",
            x,
            y,
            sensor_w,
            sensor_h
        );
        if x >= sensor_w || y >= sensor_h {
            // For safety, skip invalid inputs (release builds avoid bounds panic)
            continue;
        }

        // Normalize timestamp into [0, n_time_bins]
        let ts = (n_time_bins as f64) * ((ev.t_ns - t0) as f64) / denom;

        // Integer part and fractional remainder
        let ti_floor = ts.floor();
        let frac = ts - ti_floor;
        let mut ti = ti_floor as isize;

        // Map polarity: 0 -> -1.0; otherwise use p as f64
        let val = if ev.p == 0 { -1.0 } else { ev.p as f64 };

        // Left bin accumulation (if within range)
        if ti >= 0 && (ti as usize) < n_time_bins {
            let t = ti as usize;
            let idx = t * hw + y * sensor_w + x;
            grid[idx] += val * (1.0 - frac);
        }

        // Right bin accumulation (ti + 1)
        ti += 1;
        if ti >= 0 && (ti as usize) < n_time_bins {
            let t = ti as usize;
            let idx = t * hw + y * sensor_w + x;
            grid[idx] += val * frac;
        }
    }

    grid
}
