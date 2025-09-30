use tonic_core::kernels::time_surface::to_time_surface;
use tonic_core::Event;

#[inline]
fn flat_index(t: usize, p: usize, y: usize, x: usize, n_p: usize, h: usize, w: usize) -> usize {
    ((((t * n_p) + p) * h) + y) * w + x
}

#[inline]
fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn timesurface_shape_and_value_bounds() {
    let w = 4usize;
    let h = 3usize;
    let pch = 2usize;

    let dt: i64 = 1000;
    let tau: f64 = 2000.0;
    let overlap: i64 = 0;
    let include_incomplete = false;

    // Events spanning 3 windows: duration = 3000 -> T = duration // dt = 3
    let events = vec![
        Event { t_ns: 0,    x: 0, y: 0, p: 0, stream_id: 0 },
        Event { t_ns: 500,  x: 1, y: 1, p: 1, stream_id: 0 },
        Event { t_ns: 1500, x: 2, y: 1, p: 0, stream_id: 0 },
        Event { t_ns: 2499, x: 3, y: 2, p: 1, stream_id: 0 },
        Event { t_ns: 3000, x: 0, y: 2, p: 0, stream_id: 0 },
    ];

    let (buf, t_slices) = to_time_surface(
        &events,
        w,
        h,
        pch,
        dt,
        tau,
        overlap,
        include_incomplete,
        None,
        None,
    )
    .expect("to_time_surface should succeed");

    let duration = events.last().unwrap().t_ns - events.first().unwrap().t_ns;
    let expected_t = (duration / dt) as usize;
    assert_eq!(t_slices, expected_t);
    assert_eq!(buf.len(), t_slices * pch * h * w);

    // Values should be within [0, 1]
    assert!(
        buf.iter().all(|v| *v >= 0.0 && *v <= 1.0 + 1e-12),
        "All values must lie in [0, 1]"
    );
}

#[test]
fn timesurface_monotonic_recency_decay() {
    let w = 3usize;
    let h = 2usize;
    let pch = 2usize;

    let dt: i64 = 1000;
    let tau: f64 = 1000.0;
    let overlap: i64 = 0;

    // Two slices total: duration = 2000 -> T = 2
    // Put a single event for pixel (p=0, y=0, x=1) in the first slice at t=250.
    // Expect value at slice 1 to be value at slice 0 times exp(-dt / tau).
    let events = vec![
        Event { t_ns: 0,    x: 2, y: 1, p: 1, stream_id: 0 }, // dummy to set t_first
        Event { t_ns: 250,  x: 1, y: 0, p: 0, stream_id: 0 }, // the pixel we track
        Event { t_ns: 1000, x: 0, y: 0, p: 1, stream_id: 0 }, // dummy inside second slice
        Event { t_ns: 2000, x: 2, y: 1, p: 1, stream_id: 0 }, // dummy to set t_last
    ];

    let (buf, t_slices) = to_time_surface(
        &events,
        w,
        h,
        pch,
        dt,
        tau,
        0,
        false,
        None,
        None,
    )
    .expect("ok");

    assert_eq!(t_slices, 2, "Expected exactly two slices");

    let i0 = flat_index(0, 0, 0, 1, pch, h, w);
    let i1 = flat_index(1, 0, 0, 1, pch, h, w);

    let v0 = buf[i0];
    let v1 = buf[i1];

    assert!(v0 > 0.0 && v0 <= 1.0);
    assert!(v1 > 0.0 && v1 <= 1.0);
    let expected_ratio = (-((dt as f64) / tau)).exp();
    let ratio = v1 / v0;
    assert!(
        approx_eq(ratio, expected_ratio, 1e-12),
        "Expected ratio exp(-dt/tau) = {}, got {} (v0={}, v1={})",
        expected_ratio,
        ratio,
        v0,
        v1
    );
}

#[test]
fn timesurface_slice_count_rules_floor_ceil() {
    let w = 5usize;
    let h = 4usize;
    let pch = 2usize;

    // Create 50 events from t=0 to t=10_000 inclusive
    let n = 50usize;
    let t_last: i64 = 10_000;
    let mut events = Vec::with_capacity(n);
    for i in 0..n {
        let t = if n > 1 {
            ((i as i64) * t_last) / ((n - 1) as i64)
        } else {
            0
        };
        let x = (i % w) as u16;
        let y = ((i / w) % h) as u16;
        let p = (i % pch) as i8;
        events.push(Event { t_ns: t, x, y, p, stream_id: 0 });
    }

    let dt: i64 = 2000;
    let overlap: i64 = 500; // stride = 1500
    let stride = dt - overlap;

    // include_incomplete = false
    let (buf_f, t_f) = to_time_surface(
        &events, w, h, pch, dt, 5000.0, overlap, false, None, None,
    )
    .expect("ok (floor)");

    // include_incomplete = true
    let (buf_t, t_t) = to_time_surface(
        &events, w, h, pch, dt, 5000.0, overlap, true, None, None,
    )
    .expect("ok (ceil)");

    assert_eq!(buf_f.len(), t_f * pch * h * w);
    assert_eq!(buf_t.len(), t_t * pch * h * w);

    let t_first = events.first().unwrap().t_ns;
    let t_last_ev = events.last().unwrap().t_ns;
    let duration = t_last_ev - t_first;

    let expected_floor = (((duration - dt) as f64) / (stride as f64)).floor() as i64 + 1;
    let expected_ceil  = (((duration - dt) as f64) / (stride as f64)).ceil()  as i64 + 1;
    let expected_floor = expected_floor.max(1) as usize;
    let expected_ceil  = expected_ceil.max(1) as usize;

    assert_eq!(t_f, expected_floor, "time_window floor rule mismatch");
    assert_eq!(t_t, expected_ceil,  "time_window ceil rule mismatch");
}