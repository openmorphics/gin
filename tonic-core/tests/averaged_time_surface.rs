use tonic_core::kernels::averaged_time_surface::{to_averaged_time_surface, DecayKind};
use tonic_core::Event;

#[inline]
fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[inline]
fn flat_index(c: usize, p: usize, sy: usize, sx: usize, n_p: usize, s: usize) -> usize {
    // Layout: (C, P, S, S) row-major
    (((c * n_p) + p) * s + sy) * s + sx
}

#[inline]
fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

#[inline]
fn cell_index_for(x: usize, y: usize, cell_size: usize, wgrid: usize) -> usize {
    (y / cell_size) * wgrid + (x / cell_size)
}

#[test]
fn test_output_shape() {
    let w = 4usize;
    let h = 4usize;
    let pch = 2usize;
    let cell = 3usize;
    let s = 3usize;

    // 2x2 grid => 4 cells
    let wgrid = ceil_div(w, cell);
    let hgrid = ceil_div(h, cell);
    let n_cells = wgrid * hgrid;

    // Two events in different cells/polarities
    let events = vec![
        Event { t_ns: 0,    x: 0, y: 0, p: 0, stream_id: 0 }, // cell (0,0), p=0
        Event { t_ns: 1000, x: 3, y: 3, p: 1, stream_id: 0 }, // cell (1,1), p=1
    ];

    let buf = to_averaged_time_surface(
        &events,
        w, h, pch,
        cell,
        s,
        10_000.0, // time_window
        1_000.0,  // tau
        DecayKind::Exp,
    ).expect("ok");

    assert_eq!(buf.len(), n_cells * pch * s * s);
}

#[test]
fn test_empty_input_returns_zeros() {
    let w = 6usize;
    let h = 4usize;
    let pch = 2usize;
    let cell = 3usize;
    let s = 3usize;

    let wgrid = ceil_div(w, cell);
    let hgrid = ceil_div(h, cell);
    let n_cells = wgrid * hgrid;

    let events: Vec<Event> = vec![];
    let buf = to_averaged_time_surface(
        &events, w, h, pch, cell, s, 5_000.0, 1_000.0, DecayKind::Exp
    ).expect("ok");

    assert_eq!(buf.len(), n_cells * pch * s * s);
    assert!(buf.iter().all(|v| *v == 0.0), "Empty input must yield all zeros");
}

#[test]
fn test_single_event_handling() {
    // Single cell sensor: all events map into one cell
    let w = 3usize;
    let h = 3usize;
    let pch = 1usize;
    let cell = 3usize;
    let s = 3usize;
    let rho = s / 2;

    let events = vec![
        Event { t_ns: 42, x: 1, y: 1, p: 0, stream_id: 0 },
    ];

    let buf = to_averaged_time_surface(
        &events, w, h, pch, cell, s, 10_000.0, 1_000.0, DecayKind::Exp
    ).expect("ok");

    // Only one cell (c=0), one polarity (p=0), center = 1 after averaging by N=1
    let idx = flat_index(0, 0, rho, rho, pch, s);
    assert!(approx_eq(buf[idx], 1.0, 1e-6), "Single event should yield center value 1.0 after averaging");
    // All other entries must be zero
    for (k, v) in buf.iter().enumerate() {
        if k != idx {
            assert!(approx_eq(*v, 0.0, 1e-12));
        }
    }
}

#[test]
fn test_averaged_time_surface_basic() {
    // Two events at the same pixel within time_window should contribute exp decay at center plus 1 for the current.
    // Averaging across 2 events: expected center = (1 + (1 + exp(-dt/tau))) / 2 = 1 + 0.5 * exp(-dt/tau)
    let w = 4usize;
    let h = 4usize;
    let pch = 1usize;
    let cell = 3usize;
    let s = 3usize;
    let rho = s / 2;

    let wgrid = ceil_div(w, cell);
    let _hgrid = ceil_div(h, cell);

    // Choose a pixel in cell (0,0)
    let x = 1usize;
    let y = 1usize;

    let dt = 500i64;
    let tau = 1000.0f32;
    let exp_val = (-(dt as f32) / tau).exp();

    let events = vec![
        Event { t_ns: 1000, x: x as u16, y: y as u16, p: 0, stream_id: 0 },
        Event { t_ns: 1000 + dt, x: x as u16, y: y as u16, p: 0, stream_id: 0 },
    ];

    let buf = to_averaged_time_surface(
        &events, w, h, pch, cell, s, 1_000_000.0, tau, DecayKind::Exp
    ).expect("ok");

    let c = cell_index_for(x, y, cell, wgrid);
    let center_idx = flat_index(c, 0, rho, rho, pch, s);
    let expected_center = 1.0 + 0.5 * exp_val;
    assert!(
        approx_eq(buf[center_idx], expected_center, 1e-5),
        "center={} expected={} (exp_val={})",
        buf[center_idx], expected_center, exp_val
    );
}

#[test]
fn test_temporal_windowing() {
    // Same events as above but with a too-small time_window so previous event is excluded.
    // Then each event contributes only +1 at center; average over 2 events => 1.0 exactly.
    let w = 3usize;
    let h = 3usize;
    let pch = 1usize;
    let cell = 3usize;
    let s = 3usize;
    let rho = s / 2;

    let x = 1usize;
    let y = 1usize;

    let events = vec![
        Event { t_ns: 1000, x: x as u16, y: y as u16, p: 0, stream_id: 0 },
        Event { t_ns: 1500, x: x as u16, y: y as u16, p: 0, stream_id: 0 },
    ];

    // time_window too small to include previous (t_j >= t_i - time_window) => 1000 >= 1100 is false
    let buf = to_averaged_time_surface(
        &events, w, h, pch, cell, s, 400.0, 1_000.0, DecayKind::Exp
    ).expect("ok");

    let c = 0usize; // only one cell
    let center_idx = flat_index(c, 0, rho, rho, pch, s);
    assert!(approx_eq(buf[center_idx], 1.0, 1e-6), "center average should be 1.0 when no neighbor in window");
}

#[test]
fn test_tau_decay_averaging() {
    // Two events at same pixel included in window; smaller tau => smaller exp contribution => lower average.
    let w = 3usize;
    let h = 3usize;
    let pch = 1usize;
    let cell = 3usize;
    let s = 3usize;
    let rho = s / 2;

    let x = 1usize;
    let y = 1usize;

    let events = vec![
        Event { t_ns: 1000, x: x as u16, y: y as u16, p: 0, stream_id: 0 },
        Event { t_ns: 1500, x: x as u16, y: y as u16, p: 0, stream_id: 0 },
    ];

    let buf_small_tau = to_averaged_time_surface(
        &events, w, h, pch, cell, s, 10_000.0, 500.0, DecayKind::Exp
    ).expect("ok");
    let buf_large_tau = to_averaged_time_surface(
        &events, w, h, pch, cell, s, 10_000.0, 2000.0, DecayKind::Exp
    ).expect("ok");

    let center_idx = flat_index(0, 0, rho, rho, pch, s);
    assert!(
        buf_small_tau[center_idx] < buf_large_tau[center_idx],
        "smaller tau should produce smaller center average"
    );
}

#[test]
fn test_deterministic_output() {
    // Same input must produce identical output across runs.
    let w = 8usize;
    let h = 6usize;
    let pch = 2usize;
    let cell = 4usize;
    let s = 3usize;

    let events = vec![
        Event { t_ns: 10,  x: 2, y: 1, p: 0, stream_id: 0 },
        Event { t_ns: 20,  x: 3, y: 1, p: 1, stream_id: 0 },
        Event { t_ns: 30,  x: 7, y: 5, p: 1, stream_id: 0 },
        Event { t_ns: 40,  x: 0, y: 0, p: 0, stream_id: 0 },
        Event { t_ns: 55,  x: 4, y: 4, p: -1, stream_id: 0 }, // negative polarity -> maps to 0 channel if pch>1
        Event { t_ns: 100, x: 6, y: 5, p: 1, stream_id: 0 },
    ];

    let a = to_averaged_time_surface(
        &events, w, h, pch, cell, s, 50_000.0, 5_000.0, DecayKind::Exp
    ).expect("ok");
    let b = to_averaged_time_surface(
        &events, w, h, pch, cell, s, 50_000.0, 5_000.0, DecayKind::Exp
    ).expect("ok");

    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert!(approx_eq(a[i], b[i], 1e-12), "deterministic mismatch at {}", i);
    }
}