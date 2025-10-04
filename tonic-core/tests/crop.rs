use tonic_core::kernels::crop::{center_crop, crop_box};
use tonic_core::Event;

fn is_stable_subsequence(sub: &[Event], sup: &[Event]) -> bool {
    if sub.is_empty() {
        return true;
    }
    let mut j = 0usize;
    for e in sub {
        let mut found = false;
        while j < sup.len() {
            if *e == sup[j] {
                found = true;
                j += 1;
                break;
            }
            j += 1;
        }
        if !found {
            return false;
        }
    }
    true
}

fn is_order_preserved_via_inverse_rebase(sub: &[Event], sup: &[Event], ox: usize, oy: usize) -> bool {
    let mut last_index = 0usize;
    for e in sub {
        let orig_x = e.x as usize + ox;
        let orig_y = e.y as usize + oy;
        let mut found: Option<usize> = None;
        for i in last_index..sup.len() {
            let s = &sup[i];
            if s.t_ns == e.t_ns
                && s.x as usize == orig_x
                && s.y as usize == orig_y
                && s.p == e.p
                && s.stream_id == e.stream_id
            {
                found = Some(i);
                break;
            }
        }
        match found {
            Some(idx) => last_index = idx + 1,
            None => return false,
        }
    }
    true
}

#[test]
fn test_center_crop_basic() {
    let sensor_w = 8usize;
    let sensor_h = 6usize;
    let target_w = 4usize;
    let target_h = 2usize;

    // Center offsets: ox=(8-4)//2=2, oy=(6-2)//2=2
    // Keep region: x in [2,6), y in [2,4)
    let events = vec![
        Event { t_ns: 1, x: 1, y: 2, p:  1, stream_id: 0 }, // left of box
        Event { t_ns: 2, x: 2, y: 2, p: -1, stream_id: 0 }, // inside -> (0,0)
        Event { t_ns: 3, x: 3, y: 2, p:  1, stream_id: 0 }, // inside -> (1,0)
        Event { t_ns: 4, x: 5, y: 3, p:  1, stream_id: 0 }, // inside -> (3,1)
        Event { t_ns: 5, x: 6, y: 3, p:  1, stream_id: 0 }, // x==6 out (right edge exclusive)
        Event { t_ns: 6, x: 4, y: 4, p:  1, stream_id: 0 }, // y==4 out (bottom edge exclusive)
        Event { t_ns: 7, x: 2, y: 3, p: -1, stream_id: 0 }, // inside -> (0,1)
    ];

    let out = center_crop(&events, sensor_w, sensor_h, target_w, target_h);

    // Expected keeps in original order: indices 1,2,3,6
    assert_eq!(out.len(), 4);
    assert_eq!(out[0].x, 0); assert_eq!(out[0].y, 0); assert_eq!(out[0].p, -1);
    assert_eq!(out[1].x, 1); assert_eq!(out[1].y, 0); assert_eq!(out[1].p,  1);
    assert_eq!(out[2].x, 3); assert_eq!(out[2].y, 1); assert_eq!(out[2].p,  1);
    assert_eq!(out[3].x, 0); assert_eq!(out[3].y, 1); assert_eq!(out[3].p, -1);

    // Invariants: coordinates rebased into [0,target)
    for ev in &out {
        assert!((ev.x as usize) < target_w, "x {} not in [0,{})", ev.x, target_w);
        assert!((ev.y as usize) < target_h, "y {} not in [0,{})", ev.y, target_h);
    }

    // Order preserved relative to input after inverse rebasing
    let ox = (sensor_w - target_w) / 2;
    let oy = (sensor_h - target_h) / 2;
    assert!(is_order_preserved_via_inverse_rebase(&out, &events, ox, oy));
}

#[test]
fn test_center_crop_clamps_and_order() {
    // Target exceeds sensor; kernel clamps to sensor dims.
    let sensor_w = 5usize;
    let sensor_h = 3usize;
    let target_w = 10usize;
    let target_h = 7usize;

    let events = vec![
        Event { t_ns: 10, x: 0, y: 0, p: 1,  stream_id: 0 }, // keep
        Event { t_ns: 11, x: 4, y: 2, p: -1, stream_id: 0 }, // keep
        Event { t_ns: 12, x: 5, y: 0, p: 1,  stream_id: 0 }, // x OOB -> drop
        Event { t_ns: 13, x: 2, y: 3, p: 1,  stream_id: 0 }, // y OOB -> drop
        Event { t_ns: 14, x: 1, y: 1, p: 1,  stream_id: 0 }, // keep
    ];

    let out = center_crop(&events, sensor_w, sensor_h, target_w, target_h);

    // With clamp to (5,3), center offsets become (0,0), all in-bounds events preserved unchanged.
    assert_eq!(out.len(), 3);
    assert_eq!(out[0], events[0]);
    assert_eq!(out[1], events[1]);
    assert_eq!(out[2], events[4]);

    // Invariants and order
    let tw = target_w.min(sensor_w);
    let th = target_h.min(sensor_h);
    let ox = (sensor_w - tw) / 2;
    let oy = (sensor_h - th) / 2;
    assert!(is_order_preserved_via_inverse_rebase(&out, &events, ox, oy));
    for ev in &out {
        assert!((ev.x as usize) < sensor_w);
        assert!((ev.y as usize) < sensor_h);
    }
}

#[test]
fn test_center_crop_1d() {
    // 1D sensor: H=1. Crop horizontally to a window centered on width.
    let sensor_w = 10usize;
    let sensor_h = 1usize;
    let target_w = 4usize;
    let target_h = 1usize;

    // ox = (10 - 4) // 2 = 3; keep x in [3,7)
    let events = vec![
        Event { t_ns: 1, x: 2, y: 0, p: 1, stream_id: 0 }, // out (left)
        Event { t_ns: 2, x: 3, y: 0, p: 1, stream_id: 0 }, // in -> x'=0
        Event { t_ns: 3, x: 6, y: 0, p: 1, stream_id: 0 }, // in -> x'=3
        Event { t_ns: 4, x: 7, y: 0, p: 1, stream_id: 0 }, // out (right edge)
    ];

    let out = center_crop(&events, sensor_w, sensor_h, target_w, target_h);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].x, 0);
    assert_eq!(out[0].y, 0);
    assert_eq!(out[1].x, 3);
    assert_eq!(out[1].y, 0);

    // Invariants
    for ev in &out {
        assert!((ev.x as usize) < target_w);
        assert_eq!(ev.y, 0);
    }
    let ox = (sensor_w - target_w) / 2;
    let oy = (sensor_h - target_h) / 2;
    assert!(is_order_preserved_via_inverse_rebase(&out, &events, ox, oy));
}

#[test]
fn test_center_crop_empty_or_zero_target() {
    let sensor_w = 8usize;
    let sensor_h = 6usize;

    // Empty input
    let empty: Vec<Event> = Vec::new();
    let out_empty = center_crop(&empty, sensor_w, sensor_h, 4, 2);
    assert!(out_empty.is_empty());

    // Zero target width
    let evs = vec![Event { t_ns: 0, x: 1, y: 1, p: 1, stream_id: 0 }];
    let out_zero_w = center_crop(&evs, sensor_w, sensor_h, 0, 2);
    assert!(out_zero_w.is_empty());

    // Zero target height
    let out_zero_h = center_crop(&evs, sensor_w, sensor_h, 4, 0);
    assert!(out_zero_h.is_empty());

    // Zero sensor dims imply empty (safety)
    let out_zero_sensor = center_crop(&evs, 0, sensor_h, 2, 2);
    assert!(out_zero_sensor.is_empty());
}

#[test]
fn test_crop_box_basic() {
    let sensor_w = 8usize;
    let sensor_h = 6usize;
    let x0 = 2usize;
    let y0 = 2usize;
    let target_w = 4usize;
    let target_h = 2usize;

    // Keep region: x in [2,6), y in [2,4)
    let events = vec![
        Event { t_ns: 1, x: 1, y: 2, p:  1, stream_id: 0 }, // left of box
        Event { t_ns: 2, x: 2, y: 2, p: -1, stream_id: 0 }, // inside -> (0,0)
        Event { t_ns: 3, x: 3, y: 2, p:  1, stream_id: 0 }, // inside -> (1,0)
        Event { t_ns: 4, x: 5, y: 3, p:  1, stream_id: 0 }, // inside -> (3,1)
        Event { t_ns: 5, x: 6, y: 3, p:  1, stream_id: 0 }, // x==6 out (right edge exclusive)
        Event { t_ns: 6, x: 4, y: 4, p:  1, stream_id: 0 }, // y==4 out (bottom edge exclusive)
        Event { t_ns: 7, x: 2, y: 3, p: -1, stream_id: 0 }, // inside -> (0,1)
    ];

    let out = crop_box(&events, sensor_w, sensor_h, x0, y0, target_w, target_h);

    // Expected keeps in original order: indices 1,2,3,6
    assert_eq!(out.len(), 4);
    assert_eq!(out[0].x, 0); assert_eq!(out[0].y, 0); assert_eq!(out[0].p, -1);
    assert_eq!(out[1].x, 1); assert_eq!(out[1].y, 0); assert_eq!(out[1].p,  1);
    assert_eq!(out[2].x, 3); assert_eq!(out[2].y, 1); assert_eq!(out[2].p,  1);
    assert_eq!(out[3].x, 0); assert_eq!(out[3].y, 1); assert_eq!(out[3].p, -1);

    // Invariants: coordinates rebased into [0,target)
    for ev in &out {
        assert!((ev.x as usize) < target_w, "x {} not in [0,{})", ev.x, target_w);
        assert!((ev.y as usize) < target_h, "y {} not in [0,{})", ev.y, target_h);
    }

    // Order preserved relative to input after inverse rebasing
    assert!(is_order_preserved_via_inverse_rebase(&out, &events, x0, y0));
}

#[test]
fn test_crop_box_clamps() {
    // Target exceeds sensor; kernel effectively clamps to sensor edge.
    let sensor_w = 5usize;
    let sensor_h = 3usize;
    let x0 = 3usize;
    let y0 = 1usize;
    let target_w = 10usize;
    let target_h = 5usize;

    let events = vec![
        Event { t_ns: 10, x: 0, y: 0, p: 1,  stream_id: 0 }, // outside (left/top)
        Event { t_ns: 11, x: 4, y: 2, p: -1, stream_id: 0 }, // keep -> (1,1)
        Event { t_ns: 12, x: 3, y: 1, p: 1,  stream_id: 0 }, // keep -> (0,0)
        Event { t_ns: 13, x: 2, y: 2, p: 1,  stream_id: 0 }, // left of box -> drop
        Event { t_ns: 14, x: 4, y: 1, p: 1,  stream_id: 0 }, // keep -> (1,0)
    ];

    let out = crop_box(&events, sensor_w, sensor_h, x0, y0, target_w, target_h);

    assert_eq!(out.len(), 3);
    assert_eq!(out[0].x, 1); assert_eq!(out[0].y, 1);
    assert_eq!(out[1].x, 0); assert_eq!(out[1].y, 0);
    assert_eq!(out[2].x, 1); assert_eq!(out[2].y, 0);

    // Invariants and order under effective clamping
    let tw = target_w.min(sensor_w.saturating_sub(x0));
    let th = target_h.min(sensor_h.saturating_sub(y0));
    for ev in &out {
        assert!((ev.x as usize) < tw);
        assert!((ev.y as usize) < th);
    }
    assert!(is_order_preserved_via_inverse_rebase(&out, &events, x0, y0));
}

#[test]
fn test_crop_box_out_of_bounds_origin() {
    let sensor_w = 5usize;
    let sensor_h = 3usize;
    let events = vec![
        Event { t_ns: 1, x: 0, y: 0, p: 1, stream_id: 0 },
        Event { t_ns: 2, x: 4, y: 2, p: 1, stream_id: 0 },
    ];

    // x0 >= W
    let out1 = crop_box(&events, sensor_w, sensor_h, 5, 0, 2, 2);
    assert!(out1.is_empty());

    // y0 >= H
    let out2 = crop_box(&events, sensor_w, sensor_h, 0, 3, 2, 2);
    assert!(out2.is_empty());
}

#[test]
fn test_crop_box_1d() {
    // 1D sensor: H=1. y0 must be 0.
    let sensor_w = 10usize;
    let sensor_h = 1usize;
    let x0 = 3usize;
    let y0 = 0usize;
    let target_w = 4usize;
    let target_h = 1usize;

    let events = vec![
        Event { t_ns: 1, x: 2, y: 0, p: 1, stream_id: 0 }, // out (left)
        Event { t_ns: 2, x: 3, y: 0, p: 1, stream_id: 0 }, // in -> x'=0
        Event { t_ns: 3, x: 6, y: 0, p: 1, stream_id: 0 }, // in -> x'=3
        Event { t_ns: 4, x: 7, y: 0, p: 1, stream_id: 0 }, // out (right edge)
    ];

    let out = crop_box(&events, sensor_w, sensor_h, x0, y0, target_w, target_h);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].x, 0);
    assert_eq!(out[0].y, 0);
    assert_eq!(out[1].x, 3);
    assert_eq!(out[1].y, 0);

    // Invariants
    for ev in &out {
        assert!((ev.x as usize) < target_w);
        assert_eq!(ev.y, 0);
    }
    assert!(is_order_preserved_via_inverse_rebase(&out, &events, x0, y0));
}

#[test]
fn test_crop_box_zero_target_or_zero_sensor() {
    let sensor_w = 8usize;
    let sensor_h = 6usize;

    let evs = vec![Event { t_ns: 0, x: 1, y: 1, p: 1, stream_id: 0 }];

    // Zero target dims
    let out_zero_w = crop_box(&evs, sensor_w, sensor_h, 0, 0, 0, 2);
    assert!(out_zero_w.is_empty());
    let out_zero_h = crop_box(&evs, sensor_w, sensor_h, 0, 0, 2, 0);
    assert!(out_zero_h.is_empty());

    // Zero sensor dims imply empty
    let out_zero_sensor = crop_box(&evs, 0, sensor_h, 0, 0, 2, 2);
    assert!(out_zero_sensor.is_empty());

    // Empty input
    let empty: Vec<Event> = Vec::new();
    let out_empty = crop_box(&empty, sensor_w, sensor_h, 0, 0, 2, 2);
    assert!(out_empty.is_empty());
}