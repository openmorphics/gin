use tonic_core::{kernels::time_ops, Event};

fn ev(t: i64, x: u16, y: u16, p: i8) -> Event {
    Event { t_ns: t, x, y, p, stream_id: 0 }
}

#[test]
fn test_reflect_timestamps_over_range() {
    let input = vec![
        ev(2, 1, 1, 1),
        ev(5, 2, 2, 0),
        ev(6, 3, 4, -1),
        ev(9, 8, 7, 1),
    ];
    let t_max = input.iter().map(|e| e.t_ns).max().unwrap();
    let out = time_ops::time_reverse(&input);
    assert_eq!(out.len(), input.len(), "length must be preserved");
    for (i, src) in input.iter().enumerate() {
        let j = out.len() - 1 - i;
        let dst = &out[j];
        assert_eq!(dst.t_ns, t_max - src.t_ns, "reflected timestamp mismatch at {}", i);
        assert_eq!(dst.x, src.x, "x must be unchanged");
        assert_eq!(dst.y, src.y, "y must be unchanged");
        assert_eq!(dst.p, src.p, "p must be unchanged");
    }
    // Output timestamps should be non-decreasing if input was non-decreasing
    let mut prev = i64::MIN;
    for e in out.iter() {
        assert!(e.t_ns >= prev, "output timestamps must be non-decreasing");
        prev = e.t_ns;
    }
}

#[test]
fn test_empty_noop() {
    let input: Vec<Event> = vec![];
    let out = time_ops::time_reverse(&input);
    assert!(out.is_empty());
}

#[test]
fn test_stable_fields_and_order_mapping() {
    let input = vec![
        ev(10, 0, 0, -1),
        ev(11, 1, 2, 0),
        ev(12, 3, 4, 1),
    ];
    let t_max = 12i64;
    let out = time_ops::time_reverse(&input);
    assert_eq!(out.len(), 3);
    // Map out[k] corresponds to input[n-1-k]
    assert_eq!(out[0].x, input[2].x);
    assert_eq!(out[0].y, input[2].y);
    assert_eq!(out[0].p, input[2].p);
    assert_eq!(out[0].t_ns, t_max - input[2].t_ns);
    assert_eq!(out[1].x, input[1].x);
    assert_eq!(out[1].y, input[1].y);
    assert_eq!(out[1].p, input[1].p);
    assert_eq!(out[1].t_ns, t_max - input[1].t_ns);
    assert_eq!(out[2].x, input[0].x);
    assert_eq!(out[2].y, input[0].y);
    assert_eq!(out[2].p, input[0].p);
    assert_eq!(out[2].t_ns, t_max - input[0].t_ns);
}

#[test]
fn test_equal_timestamps_all_same() {
    let input = vec![
        ev(7, 5, 5, 1),
        ev(7, 6, 6, 0),
        ev(7, 7, 7, -1),
    ];
    let out = time_ops::time_reverse(&input);
    assert_eq!(out.len(), 3);
    // All timestamps become zero (max - t == 0)
    for e in out.iter() {
        assert_eq!(e.t_ns, 0);
    }
    // Order is reversed deterministically
    assert_eq!(out[0].x, input[2].x);
    assert_eq!(out[0].y, input[2].y);
    assert_eq!(out[0].p, input[2].p);
    assert_eq!(out[1].x, input[1].x);
    assert_eq!(out[1].y, input[1].y);
    assert_eq!(out[1].p, input[1].p);
    assert_eq!(out[2].x, input[0].x);
    assert_eq!(out[2].y, input[0].y);
    assert_eq!(out[2].p, input[0].p);
}