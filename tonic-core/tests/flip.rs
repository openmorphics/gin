use tonic_core::{kernels::flip, Event};

fn ev(t: i64, x: u16, y: u16, p: i8) -> Event {
    Event {
        t_ns: t,
        x,
        y,
        p,
        stream_id: 0,
    }
}

#[test]
fn test_flip_lr_basic() {
    // Sensor W=8, H=6
    let w = 8usize;
    let h = 6usize;

    let input = vec![
        ev(10, 0, 0, 1),
        ev(11, 3, 5, -1),
        ev(12, 7, 2, 0),
        ev(13, 1, 4, 1),
    ];

    let out = flip::flip_lr(&input, w, h);

    assert_eq!(out.len(), input.len(), "All inputs are in-bounds; length preserved.");

    for (i, (src, dst)) in input.iter().zip(out.iter()).enumerate() {
        // x' = (W-1) - x
        let expect_x = (w - 1 - src.x as usize) as u16;
        assert_eq!(dst.x, expect_x, "idx {}: x flipped incorrectly", i);
        assert_eq!(dst.y, src.y, "idx {}: y must be unchanged", i);
        assert_eq!(dst.t_ns, src.t_ns, "idx {}: t must be unchanged", i);
        assert_eq!(dst.p, src.p, "idx {}: p must be unchanged", i);
    }
}

#[test]
fn test_flip_ud_basic() {
    // Sensor W=8, H=6
    let w = 8usize;
    let h = 6usize;

    let input = vec![
        ev(10, 0, 0, 1),
        ev(11, 3, 5, -1),
        ev(12, 7, 2, 0),
        ev(13, 1, 4, 1),
    ];

    let out = flip::flip_ud(&input, w, h);

    assert_eq!(out.len(), input.len(), "All inputs are in-bounds; length preserved.");

    for (i, (src, dst)) in input.iter().zip(out.iter()).enumerate() {
        // y' = (H-1) - y
        let expect_y = (h - 1 - src.y as usize) as u16;
        assert_eq!(dst.y, expect_y, "idx {}: y flipped incorrectly", i);
        assert_eq!(dst.x, src.x, "idx {}: x must be unchanged", i);
        assert_eq!(dst.t_ns, src.t_ns, "idx {}: t must be unchanged", i);
        assert_eq!(dst.p, src.p, "idx {}: p must be unchanged", i);
    }
}

#[test]
fn test_flip_polarity_01_and_pm1() {
    // Case A: {0,1} encoding
    let input_01 = vec![
        ev(0, 1, 1, 0),
        ev(1, 2, 2, 1),
        ev(2, 3, 3, 0),
        ev(3, 4, 4, 1),
    ];
    let out_01 = flip::flip_polarity(&input_01);
    assert_eq!(out_01.len(), input_01.len());
    for (src, dst) in input_01.iter().zip(out_01.iter()) {
        let expect_p = match src.p {
            0 => 1,
            1 => 0,
            other => other,
        };
        assert_eq!(dst.p, expect_p, "01 mapping incorrect");
        assert_eq!(dst.x, src.x);
        assert_eq!(dst.y, src.y);
        assert_eq!(dst.t_ns, src.t_ns);
    }

    // Case B: {-1,1} encoding
    let input_pm1 = vec![
        ev(10, 10, 10, -1),
        ev(11, 11, 11, 1),
        ev(12, 12, 12, -1),
        ev(13, 13, 13, 1),
    ];
    let out_pm1 = flip::flip_polarity(&input_pm1);
    assert_eq!(out_pm1.len(), input_pm1.len());
    for (src, dst) in input_pm1.iter().zip(out_pm1.iter()) {
        let expect_p = match src.p {
            -1 => 1,
            1 => -1,
            other => other,
        };
        assert_eq!(dst.p, expect_p, "±1 mapping incorrect");
        assert_eq!(dst.x, src.x);
        assert_eq!(dst.y, src.y);
        assert_eq!(dst.t_ns, src.t_ns);
    }

    // Case C: Unknown encoding values (e.g., 2) left unchanged
    let input_unknown = vec![ev(100, 0, 0, 2), ev(101, 0, 0, 3)];
    let out_unknown = flip::flip_polarity(&input_unknown);
    for (src, dst) in input_unknown.iter().zip(out_unknown.iter()) {
        assert_eq!(dst.p, src.p, "Unknown polarity values must be unchanged");
        assert_eq!(dst.x, src.x);
        assert_eq!(dst.y, src.y);
        assert_eq!(dst.t_ns, src.t_ns);
    }
}

#[test]
fn test_bounds_and_empty() {
    let w = 4usize;
    let h = 3usize;

    // Mix of in-bounds and out-of-bounds (x==W or y==H are OOB)
    let input = vec![
        ev(0, 0, 0, 1),           // in
        ev(1, 3, 2, -1),          // in (max in-bounds)
        ev(2, 4, 0, 0),           // OOB x (== W)
        ev(3, 1, 3, 1),           // OOB y (== H)
        ev(4, 10, 10, 1),         // OOB both
        ev(5, 2, 1, -1),          // in
    ];

    // LR
    let out_lr = flip::flip_lr(&input, w, h);
    // Keep only the in-bounds (indices 0,1,5)
    assert_eq!(out_lr.len(), 3);
    assert_eq!(out_lr[0].t_ns, input[0].t_ns, "order must be preserved (0)");
    assert_eq!(out_lr[1].t_ns, input[1].t_ns, "order must be preserved (1)");
    assert_eq!(out_lr[2].t_ns, input[5].t_ns, "order must be preserved (5)");
    assert_eq!(out_lr[0].x, (w - 1 - input[0].x as usize) as u16);
    assert_eq!(out_lr[1].x, (w - 1 - input[1].x as usize) as u16);
    assert_eq!(out_lr[2].x, (w - 1 - input[5].x as usize) as u16);

    // UD
    let out_ud = flip::flip_ud(&input, w, h);
    assert_eq!(out_ud.len(), 3);
    assert_eq!(out_ud[0].t_ns, input[0].t_ns);
    assert_eq!(out_ud[1].t_ns, input[1].t_ns);
    assert_eq!(out_ud[2].t_ns, input[5].t_ns);
    assert_eq!(out_ud[0].y, (h - 1 - input[0].y as usize) as u16);
    assert_eq!(out_ud[1].y, (h - 1 - input[1].y as usize) as u16);
    assert_eq!(out_ud[2].y, (h - 1 - input[5].y as usize) as u16);

    // Empty input
    let empty: Vec<Event> = vec![];
    let out_empty_lr = flip::flip_lr(&empty, w, h);
    let out_empty_ud = flip::flip_ud(&empty, w, h);
    let out_empty_pol = flip::flip_polarity(&empty);
    assert!(out_empty_lr.is_empty());
    assert!(out_empty_ud.is_empty());
    assert!(out_empty_pol.is_empty());
}

#[test]
fn test_1d_sensor_ud_noop_and_lr_ok() {
    // 1D in y (H=1) => UD should be a no-op on y==0
    let w = 5usize;
    let h = 1usize;
    let input = vec![
        ev(0, 0, 0, 1),
        ev(1, 2, 0, -1),
        ev(2, 4, 0, 1),
    ];

    // UD: y' = (H-1) - y = 0 - y = 0
    let out_ud = flip::flip_ud(&input, w, h);
    assert_eq!(out_ud.len(), input.len());
    for (src, dst) in input.iter().zip(out_ud.iter()) {
        assert_eq!(dst.y, 0, "UD must be no-op for H==1");
        assert_eq!(dst.x, src.x);
        assert_eq!(dst.t_ns, src.t_ns);
        assert_eq!(dst.p, src.p);
    }

    // LR should still work normally
    let out_lr = flip::flip_lr(&input, w, h);
    assert_eq!(out_lr.len(), input.len());
    for (src, dst) in input.iter().zip(out_lr.iter()) {
        let expect_x = (w - 1 - src.x as usize) as u16;
        assert_eq!(dst.x, expect_x, "LR must flip x even when H==1");
        assert_eq!(dst.y, 0);
        assert_eq!(dst.t_ns, src.t_ns);
        assert_eq!(dst.p, src.p);
    }
}