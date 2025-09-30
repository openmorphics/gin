use crate::events::Event;
use crate::slicers::slice_by_time;

/// Compute flattened index into (T, P, H, W) row-major (C-order)
#[inline]
fn flat_index(t: usize, p: usize, y: usize, x: usize, n_p: usize, h: usize, w: usize) -> usize {
    ((((t * n_p) + p) * h) + y) * w + x
}

/// Time-surface kernel (time-window slicing).
///
/// Semantics:
/// - Uses time-window slicing with parameters (dt, overlap, include_incomplete, start_time, end_time)
/// - Maintains last-event timestamp per (polarity, y, x) across slices
/// - For slice i, current_time = start_of_first_window + (i+1) * dt
/// - Value at (p,y,x) = exp(-(current_time - last_t[p,y,x]) / tau) if last_t is set; else 0.0
///
/// Layout:
/// - Output buffer is contiguous f64 with layout (T, P, H, W) flattened in row-major (C-order).
///
/// Edge cases:
/// - If dt <= 0: returns Err("Parameter delta_t cannot be negative.")
/// - If any of sensor_w/h or n_polarities is zero: returns (Vec::new(), 0)
/// - If events are empty or slicing yields zero windows: returns (Vec::new(), 0)
pub fn to_time_surface(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
    dt: i64,
    tau: f64,
    overlap: i64,
    include_incomplete: bool,
    start_time: Option<i64>,
    end_time: Option<i64>,
) -> Result<(Vec<f64>, usize), String> {
    if sensor_w == 0 || sensor_h == 0 || n_polarities == 0 {
        return Ok((Vec::new(), 0));
    }
    if dt <= 0 {
        return Err("Parameter delta_t cannot be negative.".to_string());
    }
    if events.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let t_first = events.first().unwrap().t_ns;
    let t_last = events.last().unwrap().t_ns;
    let times: Vec<i64> = events.iter().map(|e| e.t_ns).collect();

    let slices = slice_by_time(
        t_first,
        t_last,
        dt,
        overlap,
        include_incomplete,
        start_time,
        end_time,
        &times,
    );
    let t_slices = slices.len();
    if t_slices == 0 {
        return Ok((Vec::new(), 0));
    }

    let start_of_first_window = start_time.unwrap_or(t_first);

    let hw = sensor_h * sensor_w;
    let phw = n_polarities * hw;
    let total_len = t_slices * phw;

    // Last event time memory initialized to sentinel (unset)
    let mut last_t = vec![i64::MIN; phw];

    let mut out = vec![0.0f64; total_len];

    for (i, (lo, hi)) in slices.iter().enumerate() {
        // Update last_t with max timestamp per (p,y,x) for events within this slice
        if *lo < *hi && *hi <= events.len() {
            for ev in &events[*lo..*hi] {
                let x = ev.x as usize;
                let y = ev.y as usize;
                if x >= sensor_w || y >= sensor_h {
                    continue;
                }

                // Polarity channel mapping:
                // - If n_polarities == 1, force p_idx = 0
                // - Else, use event p if in [0, n_polarities); skip otherwise (incl. negatives)
                let p_idx = if n_polarities == 1 {
                    0usize
                } else {
                    if ev.p < 0 {
                        continue;
                    }
                    let p_usize = ev.p as usize;
                    if p_usize >= n_polarities {
                        continue;
                    }
                    p_usize
                };

                let idx = p_idx * hw + y * sensor_w + x;
                let t_ev = ev.t_ns;
                // Use maximum timestamp within the slice for parity with vectorized Python behavior
                if t_ev > last_t[idx] {
                    last_t[idx] = t_ev;
                }
            }
        }

        // Compute the surface for this slice for all pixels/channels.
        let current_time = start_of_first_window + ((i as i64) + 1) * dt;
        let base_t = i * phw;

        // Avoid division by zero: Python does not assert on tau; assume tau > 0 in practice.
        // If tau == 0, exp(-inf/0) is undefined; here we'll let Rust produce inf/NaN accordingly.
        for p in 0..n_polarities {
            for y in 0..sensor_h {
                for x in 0..sensor_w {
                    let idx = p * hw + y * sensor_w + x;
                    let out_idx = base_t + idx;
                    let lt = last_t[idx];
                    if lt == i64::MIN {
                        out[out_idx] = 0.0;
                    } else {
                        let diff = (current_time - lt) as f64;
                        out[out_idx] = (-diff / tau).exp();
                    }
                }
            }
        }
    }

    Ok((out, t_slices))
}