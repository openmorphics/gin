// Slicers: helper functions to compute event index windows based on time/count/bin policies.
// Assumptions:
// - event_times is sorted ascending
// - (lo, hi) pairs are half-open [lo, hi)
// - semantics mirror Python slicers (tonic.slicers) where applicable

fn lower_bound(times: &[i64], target: i64) -> usize {
    // First index i where times[i] >= target
    match times.binary_search_by(|probe| {
        if *probe < target {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    }) {
        Ok(_) => unreachable!("binary_search_by with custom comparator should not return Ok"),
        Err(idx) => idx,
    }
}

/// Time window slicing:
/// - stride = time_window - overlap (must be > 0)
/// - start = start_time.unwrap_or(t_first)
/// - end   = end_time.unwrap_or(t_last)
/// - windows are [w_start, w_start + time_window)
/// - include_incomplete toggles ceil/floor for trailing window
///
/// If event_times is empty, returns empty slice list.
pub fn slice_by_time(
    t_first: i64,
    t_last: i64,
    time_window: i64,
    overlap: i64,
    include_incomplete: bool,
    start_time: Option<i64>,
    end_time: Option<i64>,
    event_times: &[i64],
) -> Vec<(usize, usize)> {
    let stride = time_window - overlap;
    assert!(stride > 0, "Inferred stride <= 0 for time window slicer");

    if event_times.is_empty() {
        return vec![];
    }

    let start = start_time.unwrap_or(t_first);
    let end = end_time.unwrap_or(t_last);
    let duration = end - start;

    // Number of slices
    let n_float = (duration - time_window) as f64 / (stride as f64);
    let mut n_slices = if include_incomplete {
        n_float.ceil()
    } else {
        n_float.floor()
    } + 1.0;
    if n_slices < 1.0 {
        n_slices = 1.0; // match Python's behavior for long strides/short recordings
    }
    let n_slices = n_slices as i64;

    let mut out = Vec::with_capacity(n_slices as usize);
    for i in 0..n_slices {
        let w_start = start + i * stride;
        let w_end = w_start + time_window;
        let lo = lower_bound(event_times, w_start);
        let hi = lower_bound(event_times, w_end);
        out.push((lo, hi));
    }
    out
}

/// Event count slicing:
/// - stride = count - overlap (must be > 0)
/// - n_slices = floor/ceil depending on include_incomplete using:
///     if include_incomplete:
///         ceil((n_events - event_count) / stride) + 1
///     else:
///         floor((n_events - event_count) / stride) + 1
/// - returns contiguous [start, end) spans
///
/// Mirrors Python's SliceByEventCount semantics. If n_events == 0, this returns one empty slice (0, 0).
pub fn slice_by_count(
    n_events: usize,
    count: usize,
    overlap: usize,
    include_incomplete: bool,
) -> Vec<(usize, usize)> {
    let stride = count as isize - overlap as isize;
    assert!(stride > 0, "Inferred stride <= 0 for count slicer");
    let stride = stride as usize;

    let event_count = count.min(n_events);

    // Note: mirrors Python including edge case n_events == 0 -> one empty slice.
    let base = if n_events >= event_count {
        n_events - event_count
    } else {
        0
    };
    let n_float = base as f64 / stride as f64;
    let n_slices = if include_incomplete {
        n_float.ceil() as usize + 1
    } else {
        n_float.floor() as usize + 1
    };

    let mut out = Vec::with_capacity(n_slices);
    for i in 0..n_slices {
        let start = i * stride;
        let end = start + event_count;
        out.push((start.min(n_events), end.min(n_events)));
    }
    out
}

/// Time bins slicing:
/// - duration = t_last - t_first
/// - base = duration / bin_count
/// - time_window = floor(base * (1 + overlap_frac))
/// - stride = floor(time_window * (1 - overlap_frac))
/// - produce exactly bin_count slices starting at t_first with given stride.
pub fn slice_by_time_bins(
    t_first: i64,
    t_last: i64,
    bin_count: usize,
    overlap_frac: f64,
    event_times: &[i64],
) -> (Vec<(usize, usize)>, i64, i64) {
    assert!(overlap_frac < 1.0, "overlap_frac must be < 1.0");
    if bin_count == 0 {
        return (vec![], 0, 0);
    }
    let duration = t_last - t_first;
    let base = if bin_count > 0 { duration / bin_count as i64 } else { 0 };
    let time_window = ((base as f64) * (1.0 + overlap_frac)).floor() as i64;
    let stride = ((time_window as f64) * (1.0 - overlap_frac)).floor() as i64;

    let mut out = Vec::with_capacity(bin_count);
    for i in 0..bin_count {
        let w_start = t_first + (i as i64) * stride;
        let w_end = w_start + time_window;
        let lo = lower_bound(event_times, w_start);
        let hi = lower_bound(event_times, w_end);
        out.push((lo, hi));
    }

    (out, time_window, stride)
}

/// Event bins slicing:
/// - base = n_events // bin_count
/// - spike_count = floor(base * (1 + overlap_frac))
/// - stride      = floor(spike_count * (1 - overlap_frac))
/// - produce exactly bin_count slices [start, start + spike_count)
pub fn slice_by_event_bins(
    n_events: usize,
    bin_count: usize,
    overlap_frac: f64,
) -> (Vec<(usize, usize)>, usize, usize) {
    assert!(overlap_frac < 1.0, "overlap_frac must be < 1.0");
    if bin_count == 0 {
        return (vec![], 0, 0);
    }
    let base = if bin_count > 0 { n_events / bin_count } else { 0 };
    let spike_count = ((base as f64) * (1.0 + overlap_frac)).floor() as usize;
    let stride = ((spike_count as f64) * (1.0 - overlap_frac)).floor() as usize;

    let mut out = Vec::with_capacity(bin_count);
    for i in 0..bin_count {
        let start = i * stride;
        let end = start + spike_count;
        out.push((start.min(n_events), end.min(n_events)));
    }
    (out, spike_count, stride)
}