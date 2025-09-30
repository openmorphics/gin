use crate::events::Event;
use crate::slicers::{slice_by_count, slice_by_event_bins, slice_by_time, slice_by_time_bins};

/// Compute flattened index into (T, P, H, W) row-major (C-order)
#[inline]
fn flat_index(t: usize, p: usize, y: usize, x: usize, n_p: usize, h: usize, w: usize) -> usize {
    ((((t * n_p) + p) * h) + y) * w + x
}

#[inline]
fn total_len(t: usize, p: usize, h: usize, w: usize) -> usize {
    t.saturating_mul(p).saturating_mul(h).saturating_mul(w)
}

/// Enforce the single-polarity rule (Python ValueError behavior).
/// When n_polarities == 1, if the input events contain more than one unique polarity value, return an Err.
fn check_single_polarity_rule(events: &[Event], n_polarities: usize) -> Result<(), String> {
    if n_polarities == 1 {
        let mut seen: Option<i8> = None;
        for ev in events {
            match seen {
                None => seen = Some(ev.p),
                Some(p0) => {
                    if ev.p != p0 {
                        return Err("Single polarity sensor, but events contain both polarities.".to_string());
                    }
                }
            }
        }
    }
    Ok(())
}

/// Accumulate counts per (time-slice, polarity, y, x) into an i32 buffer.
/// - P index mapping:
///   - If n_polarities == 1, p_idx is forced to 0.
///   - Otherwise, use the event's p directly as a channel index if it is within [0, n_polarities).
///     Out-of-range polarity values are skipped.
/// - Bounds: skip out-of-range x/y to avoid panics.
/// - Each event contributes +1 to its corresponding bin.
fn accumulate_into(
    buf: &mut [i32],
    events: &[Event],
    slices: &[(usize, usize)],
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
) {
    let hw = sensor_h * sensor_w;
    let phw = n_polarities * hw;

    for (t_idx, (lo, hi)) in slices.iter().enumerate() {
        if *lo >= *hi || *hi > events.len() {
            continue;
        }
        let slice_events = &events[*lo..*hi];
        for ev in slice_events {
            let x = ev.x as usize;
            let y = ev.y as usize;
            if x >= sensor_w || y >= sensor_h {
                continue;
            }

            let p_idx = if n_polarities == 1 {
                0usize
            } else {
                // Do not remap values; skip if out of range.
                if ev.p < 0 {
                    continue;
                }
                let p_usize = ev.p as usize;
                if p_usize >= n_polarities {
                    continue;
                }
                p_usize
            };

            let base_t = t_idx * phw;
            let base_tp = base_t + p_idx * hw;
            let idx = base_tp + y * sensor_w + x;
            // Safe: idx is within buf because buf length = T * P * H * W
            buf[idx] += 1;
        }
    }
}

/// Frame kernel (time-window slicing).
///
/// - Produces dense raster counts in i16 with layout (T, P, H, W) flattened in row-major order.
/// - Accumulates in i32 then casts to i16 on return.
/// - If n_polarities == 1 and events contain multiple distinct p values, returns Err.
pub fn to_frame_time_window(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
    time_window: i64,
    overlap: i64,
    include_incomplete: bool,
    start_time: Option<i64>,
    end_time: Option<i64>,
) -> Result<(Vec<i16>, usize), String> {
    if sensor_w == 0 || sensor_h == 0 || n_polarities == 0 {
        return Ok((Vec::new(), 0));
    }

    // Empty input: zero slices as per spec.
    if events.is_empty() {
        return Ok((Vec::new(), 0));
    }

    check_single_polarity_rule(events, n_polarities)?;

    let t_first = events.first().unwrap().t_ns;
    let t_last = events.last().unwrap().t_ns;
    let times: Vec<i64> = events.iter().map(|e| e.t_ns).collect();

    let slices = slice_by_time(
        t_first,
        t_last,
        time_window,
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

    let mut buf32 = vec![0i32; total_len(t_slices, n_polarities, sensor_h, sensor_w)];
    accumulate_into(&mut buf32, events, &slices, sensor_w, sensor_h, n_polarities);

    // Cast to i16 on store
    let buf16: Vec<i16> = buf32.into_iter().map(|v| v as i16).collect();
    Ok((buf16, t_slices))
}

/// Frame kernel (event-count slicing).
///
/// - Produces dense raster counts in i16 with layout (T, P, H, W) flattened in row-major order.
/// - Accumulates in i32 then casts to i16 on return.
/// - If n_polarities == 1 and events contain multiple distinct p values, returns Err.
/// - Empty inputs yield zero slices (buffer length 0).
pub fn to_frame_event_count(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
    event_count: usize,
    overlap: usize,
    include_incomplete: bool,
) -> Result<(Vec<i16>, usize), String> {
    if sensor_w == 0 || sensor_h == 0 || n_polarities == 0 {
        return Ok((Vec::new(), 0));
    }

    if events.is_empty() {
        return Ok((Vec::new(), 0));
    }

    check_single_polarity_rule(events, n_polarities)?;

    let n_events = events.len();
    let slices = slice_by_count(n_events, event_count, overlap, include_incomplete);
    let t_slices = slices.len();
    if t_slices == 0 {
        return Ok((Vec::new(), 0));
    }

    let mut buf32 = vec![0i32; total_len(t_slices, n_polarities, sensor_h, sensor_w)];
    accumulate_into(&mut buf32, events, &slices, sensor_w, sensor_h, n_polarities);

    let buf16: Vec<i16> = buf32.into_iter().map(|v| v as i16).collect();
    Ok((buf16, t_slices))
}

/// Frame kernel (fixed number of time bins).
///
/// - Always produces exactly n_time_bins slices (T = n_time_bins), even if events are empty.
/// - Overlap specified as a fraction of a bin length; time_window and stride derived as in Python.
/// - Accumulates in i32 then casts to i16 on return.
/// - If n_polarities == 1 and events contain multiple distinct p values, returns Err.
pub fn to_frame_n_time_bins(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
    n_time_bins: usize,
    overlap_frac: f64,
) -> Result<(Vec<i16>, usize), String> {
    if sensor_w == 0 || sensor_h == 0 || n_polarities == 0 || n_time_bins == 0 {
        return Ok((Vec::new(), 0));
    }

    // If empty: produce T == n_time_bins zero-filled frames.
    if events.is_empty() {
        let len = total_len(n_time_bins, n_polarities, sensor_h, sensor_w);
        return Ok((vec![0i16; len], n_time_bins));
    }

    check_single_polarity_rule(events, n_polarities)?;

    let t_first = events.first().unwrap().t_ns;
    let t_last = events.last().unwrap().t_ns;
    let times: Vec<i64> = events.iter().map(|e| e.t_ns).collect();

    let (slices, _tw, _stride) = slice_by_time_bins(t_first, t_last, n_time_bins, overlap_frac, &times);
    let t_slices = slices.len(); // should equal n_time_bins
    let mut buf32 = vec![0i32; total_len(t_slices, n_polarities, sensor_h, sensor_w)];
    accumulate_into(&mut buf32, events, &slices, sensor_w, sensor_h, n_polarities);

    let buf16: Vec<i16> = buf32.into_iter().map(|v| v as i16).collect();
    Ok((buf16, t_slices))
}

/// Frame kernel (fixed number of event bins).
///
/// - Always produces exactly n_event_bins slices (T = n_event_bins), even if events are empty.
/// - Overlap specified as a fraction of a bin count; spike_count and stride derived as in Python.
/// - Accumulates in i32 then casts to i16 on return.
/// - If n_polarities == 1 and events contain multiple distinct p values, returns Err.
pub fn to_frame_n_event_bins(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
    n_event_bins: usize,
    overlap_frac: f64,
) -> Result<(Vec<i16>, usize), String> {
    if sensor_w == 0 || sensor_h == 0 || n_polarities == 0 || n_event_bins == 0 {
        return Ok((Vec::new(), 0));
    }

    if events.is_empty() {
        let len = total_len(n_event_bins, n_polarities, sensor_h, sensor_w);
        return Ok((vec![0i16; len], n_event_bins));
    }

    check_single_polarity_rule(events, n_polarities)?;

    let n_events = events.len();
    let (slices, _spike_count, _stride) = slice_by_event_bins(n_events, n_event_bins, overlap_frac);
    let t_slices = slices.len(); // should equal n_event_bins

    let mut buf32 = vec![0i32; total_len(t_slices, n_polarities, sensor_h, sensor_w)];
    accumulate_into(&mut buf32, events, &slices, sensor_w, sensor_h, n_polarities);

    let buf16: Vec<i16> = buf32.into_iter().map(|v| v as i16).collect();
    Ok((buf16, t_slices))
}