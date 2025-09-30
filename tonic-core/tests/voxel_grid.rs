use tonic_core::kernels::voxel_grid::to_voxel_grid;
use tonic_core::Event;

fn idx(t: usize, w: usize, h: usize, x: usize, y: usize) -> usize {
    t * (h * w) + y * w + x
}

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn empty_events_returns_zeroed_buffer_and_correct_length() {
    let w = 4usize;
    let h = 3usize;
    let t = 5usize;
    let events: Vec<Event> = vec![];

    let out = to_voxel_grid(&events, w, h, t);
    assert_eq!(out.len(), t * h * w);

    // spot-check a few entries are zero
    assert!(out.iter().all(|v| *v == 0.0));
}

#[test]
fn sum_over_time_for_pixel_matches_polarity_sum() {
    // Two events at the same pixel across the time window: +1 and 0 (mapped to -1) -> sum 0
    let w = 4usize;
    let h = 3usize;
    let t_bins = 5usize;

    // Time window [100, 200] enforced by adding a dummy event at t=200 on another pixel.
    // This mirrors Python's normalization using first/last event timestamps.
    let events = vec![
        Event { t_ns: 100, x: 1, y: 2, p:  1, stream_id: 0 },
        Event { t_ns: 175, x: 1, y: 2, p:  0, stream_id: 0 }, // mapped to -1
        Event { t_ns: 200, x: 0, y: 0, p:  1, stream_id: 0 }, // dummy to set t_last
    ];

    let out = to_voxel_grid(&events, w, h, t_bins);

    let mut sum = 0.0f64;
    for t in 0..t_bins {
        sum += out[idx(t, w, h, 1, 2)];
    }
    assert!(approx_eq(sum, 0.0, 1e-12), "sum over time should be 0, got {}", sum);
}

#[test]
fn bilinear_time_accumulation_splits_adjacent_bins_correctly() {
    // Window [100,200]; bins=5, so normalized time ts = 5*(t-100)/100.
    // Event A: t=125 -> ts=1.25, contributes 0.75 to bin 1 and 0.25 to bin 2.
    // Event B: t=175 (p=0 -> -1) -> ts=3.75, contributes -0.25 to bin 3 and -0.75 to bin 4.
    let w = 4usize;
    let h = 3usize;
    let t_bins = 5usize;

    // Include dummies at t=100 (t0) and t=200 (t_last) on a different pixel to define the window.
    let events = vec![
        Event { t_ns: 100, x: 0, y: 0, p:  1, stream_id: 0 }, // dummy to set t0
        Event { t_ns: 125, x: 1, y: 2, p:  1, stream_id: 0 },
        Event { t_ns: 175, x: 1, y: 2, p:  0, stream_id: 0 }, // mapped to -1
        Event { t_ns: 200, x: 0, y: 0, p:  1, stream_id: 0 }, // dummy to set t_last
    ];

    let out = to_voxel_grid(&events, w, h, t_bins);

    let a_bin1 = out[idx(1, w, h, 1, 2)];
    let a_bin2 = out[idx(2, w, h, 1, 2)];
    let b_bin3 = out[idx(3, w, h, 1, 2)];
    let b_bin4 = out[idx(4, w, h, 1, 2)];

    // Expected contributions
    assert!(approx_eq(a_bin1, 0.75, 1e-12), "bin1 expected 0.75, got {}", a_bin1);
    assert!(approx_eq(a_bin2, 0.25, 1e-12), "bin2 expected 0.25, got {}", a_bin2);
    assert!(approx_eq(b_bin3, -0.25, 1e-12), "bin3 expected -0.25, got {}", b_bin3);
    assert!(approx_eq(b_bin4, -0.75, 1e-12), "bin4 expected -0.75, got {}", b_bin4);

    // Spot-check unrelated pixel remains zero
    // Use a pixel not touched by dummy events to verify unrelated locations remain zero.
    assert!(approx_eq(out[idx(0, w, h, 3, 0)], 0.0, 1e-12));
}
