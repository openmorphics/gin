use tonic_core::kernels::frame::{
    to_frame_event_count, to_frame_n_event_bins, to_frame_n_time_bins, to_frame_time_window,
};
use tonic_core::Event;

fn per_slice_len(p: usize, h: usize, w: usize) -> usize {
    p * h * w
}

fn sum_slice_i16(buf: &[i16], t: usize, p: usize, h: usize, w: usize) -> i64 {
    let ps = per_slice_len(p, h, w);
    let start = t * ps;
    let end = start + ps;
    buf[start..end].iter().map(|v| *v as i64).sum()
}

fn make_uniform_events_2d(n: usize, w: usize, h: usize, t_last: i64, polys: &[i8]) -> Vec<Event> {
    // Generate n events with timestamps approximately uniformly distributed in [0, t_last],
    // x cycling across width, y cycling across height, polarity cycling across given polys.
    // Timestamps: floor(i * t_last / (n-1)) for i in [0..n), which ensures t[-1]-t[0] == t_last.
    let mut events = Vec::with_capacity(n);
    for i in 0..n {
        let t = if n > 1 {
            ((i as i64) * t_last) / ((n - 1) as i64)
        } else {
            0
        };
        let x = (i % w) as u16;
        let y = ((i / w) % h) as u16;
        let p = polys[i % polys.len()];
        events.push(Event { t_ns: t, x, y, p, stream_id: 0 });
    }
    events
}

#[test]
fn event_count_slicing_shapes_and_sums() {
    let w = 8usize;
    let h = 6usize;
    let p = 2usize;
    let n_events = 50usize;
    let polys = [0i8, 1i8];

    // timestamps don't matter for count slicing; set t_last simple
    let events = make_uniform_events_2d(n_events, w, h, 1000, &polys);

    let event_count = 10usize;
    let overlap = 2usize; // stride = 8
    let include_incomplete = false;

    let (buf, t_slices) = to_frame_event_count(
        &events,
        w,
        h,
        p,
        event_count,
        overlap,
        include_incomplete,
    )
    .expect("should succeed");

    // Expected number of slices
    let stride = event_count - overlap;
    let expected_t = ((n_events - event_count) as f64 / stride as f64).floor() as usize + 1;
    assert_eq!(t_slices, expected_t, "T mismatch for event_count slicing");

    // Buffer length equals T * P * H * W
    assert_eq!(buf.len(), t_slices * p * h * w);

    // First slice is complete -> sum equals event_count
    let first_sum = sum_slice_i16(&buf, 0, p, h, w);
    assert_eq!(first_sum as usize, event_count);
}

#[test]
fn time_window_slicing_slice_count_matches_formulas() {
    let w = 10usize;
    let h = 5usize;
    let p = 2usize;
    let n_events = 200usize;
    let polys = [0i8, 1i8];
    // Set t_last = 10_000 for clean arithmetic
    let t_last: i64 = 10_000;
    let events = make_uniform_events_2d(n_events, w, h, t_last, &polys);

    // Use time_window and overlap with include_incomplete=false and true and compare T
    let time_window = 2000i64;
    let overlap = 500i64; // stride = 1500
    let start_time = None;
    let end_time = None;

    let (buf_f, t_f) = to_frame_time_window(
        &events,
        w,
        h,
        p,
        time_window,
        overlap,
        false,
        start_time,
        end_time,
    )
    .expect("ok");
    assert_eq!(buf_f.len(), t_f * p * h * w);

    let (buf_t, t_t) = to_frame_time_window(
        &events,
        w,
        h,
        p,
        time_window,
        overlap,
        true,
        start_time,
        end_time,
    )
    .expect("ok");
    assert_eq!(buf_t.len(), t_t * p * h * w);

    let stride = time_window - overlap;
    let t_first = events.first().unwrap().t_ns;
    let t_last_ev = events.last().unwrap().t_ns;
    let duration = t_last_ev - t_first;

    let expected_floor = ((duration - time_window) as f64 / stride as f64).floor() as i64 + 1;
    let expected_ceil = ((duration - time_window) as f64 / stride as f64).ceil() as i64 + 1;
    let expected_floor = expected_floor.max(1) as usize;
    let expected_ceil = expected_ceil.max(1) as usize;

    assert_eq!(t_f, expected_floor, "time_window floor rule mismatch");
    assert_eq!(t_t, expected_ceil, "time_window ceil rule mismatch");
}

#[test]
fn time_bins_fixed_count_and_overlap_distribution_first_slice_sum() {
    let w = 12usize;
    let h = 7usize;
    let p = 2usize;
    let n_events = 100usize;
    let polys = [0i8, 1i8];

    // Choose t_last so that base = duration/bin_count equals n_events/bin_count exactly.
    let n_time_bins = 5usize;
    let t_last: i64 = n_time_bins as i64 * (n_events as i64 / n_time_bins as i64);
    let events = make_uniform_events_2d(n_events, w, h, t_last, &polys);

    // No overlap
    let (buf0, t0) = to_frame_n_time_bins(&events, w, h, p, n_time_bins, 0.0).expect("ok");
    assert_eq!(t0, n_time_bins);
    assert_eq!(buf0.len(), n_time_bins * p * h * w);

    // First slice sum should equal floor(n_events / n_time_bins) for uniform distribution and 0 overlap
    let per_bin = (n_events / n_time_bins) as i64;
    let s0 = sum_slice_i16(&buf0, 0, p, h, w);
    assert_eq!(s0, per_bin);

    // With 0.1 overlap: expected approximately floor((1+overlap) * per_bin)
    let overlap = 0.1f64;
    let (buf1, t1) = to_frame_n_time_bins(&events, w, h, p, n_time_bins, overlap).expect("ok");
    assert_eq!(t1, n_time_bins);
    let expected_first = ((per_bin as f64) * (1.0 + overlap)).floor() as i64;
    let s1 = sum_slice_i16(&buf1, 0, p, h, w);
    assert_eq!(s1, expected_first);
}

#[test]
fn event_bins_fixed_count_and_first_slice_sum() {
    let w = 10usize;
    let h = 4usize;
    let p = 2usize;
    let n_events = 100usize;
    let polys = [0i8, 1i8];

    let events = make_uniform_events_2d(n_events, w, h, 5000, &polys);
    let n_event_bins = 5usize;

    // No overlap
    let (buf0, t0) =
        to_frame_n_event_bins(&events, w, h, p, n_event_bins, 0.0).expect("ok");
    assert_eq!(t0, n_event_bins);
    assert_eq!(buf0.len(), n_event_bins * p * h * w);
    // First slice sum equals base
    let base = (n_events / n_event_bins) as i64;
    let s0 = sum_slice_i16(&buf0, 0, p, h, w);
    assert_eq!(s0, base);

    // Overlap 0.25
    let overlap = 0.25f64;
    let (buf1, t1) =
        to_frame_n_event_bins(&events, w, h, p, n_event_bins, overlap).expect("ok");
    assert_eq!(t1, n_event_bins);
    let expected = ((base as f64) * (1.0 + overlap)).floor() as i64;
    let s1 = sum_slice_i16(&buf1, 0, p, h, w);
    assert_eq!(s1, expected);
}

#[test]
fn single_polarity_rule_errors_when_mixed_polarities_and_p_equals_one() {
    let w = 6usize;
    let h = 5usize;
    let p_channels = 1usize; // single polarity output requested

    // Create two events with different polarities
    let events = vec![
        Event { t_ns: 0, x: 1, y: 0, p: 0, stream_id: 0 },
        Event { t_ns: 10, x: 2, y: 0, p: 1, stream_id: 0 },
    ];

    // time window
    let err1 = to_frame_time_window(&events, w, h, p_channels, 5, 0, false, None, None);
    assert!(err1.is_err());

    // event count
    let err2 = to_frame_event_count(&events, w, h, p_channels, 2, 0, false);
    assert!(err2.is_err());

    // time bins (T fixed; still should error before accumulation)
    let err3 = to_frame_n_time_bins(&events, w, h, p_channels, 3, 0.0);
    assert!(err3.is_err());

    // event bins (T fixed; still should error)
    let err4 = to_frame_n_event_bins(&events, w, h, p_channels, 3, 0.0);
    assert!(err4.is_err());
}

#[test]
fn one_dimensional_sensor_h_equals_one_layout_and_counts() {
    let w = 16usize;
    let h = 1usize;
    let p = 2usize;
    let n_events = 32usize;
    let polys = [0i8, 1i8];

    // All events at y=0
    let mut events = Vec::with_capacity(n_events);
    for i in 0..n_events {
        let t = i as i64;
        let x = (i % w) as u16;
        let y = 0u16;
        let pol = polys[i % polys.len()];
        events.push(Event { t_ns: t, x, y, p: pol, stream_id: 0 });
    }

    // Use event bins (fixed T)
    let n_bins = 4usize;
    let (buf, t_slices) =
        to_frame_n_event_bins(&events, w, h, p, n_bins, 0.0).expect("ok");

    // Shape: T * P * H * W but H == 1
    assert_eq!(t_slices, n_bins);
    assert_eq!(buf.len(), n_bins * p * h * w);

    // First slice sum equals base
    let base = (n_events / n_bins) as i64;
    let s0 = sum_slice_i16(&buf, 0, p, h, w);
    assert_eq!(s0, base);
}

#[test]
fn empty_inputs_behavior() {
    let w = 10usize;
    let h = 5usize;
    let p = 2usize;
    let events: Vec<Event> = vec![];

    // time-window and event-count => zero T and empty buffer per spec
    let (buf_tw, t_tw) = to_frame_time_window(&events, w, h, p, 100, 0, false, None, None).unwrap();
    assert_eq!(t_tw, 0);
    assert!(buf_tw.is_empty());

    let (buf_ec, t_ec) = to_frame_event_count(&events, w, h, p, 10, 0, false).unwrap();
    assert_eq!(t_ec, 0);
    assert!(buf_ec.is_empty());

    // fixed-bin variants always produce T slices, zero-filled
    let (buf_tb, t_tb) = to_frame_n_time_bins(&events, w, h, p, 3, 0.0).unwrap();
    assert_eq!(t_tb, 3);
    assert_eq!(buf_tb.len(), 3 * p * h * w);
    assert!(buf_tb.iter().all(|v| *v == 0));

    let (buf_eb, t_eb) = to_frame_n_event_bins(&events, w, h, p, 4, 0.0).unwrap();
    assert_eq!(t_eb, 4);
    assert_eq!(buf_eb.len(), 4 * p * h * w);
    assert!(buf_eb.iter().all(|v| *v == 0));
}