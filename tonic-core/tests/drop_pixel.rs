use tonic_core::kernels::drop_ops::{identify_hot_pixel, drop_pixel};
use tonic_core::Event;

fn is_stable_subsequence(sub: &[Event], sup: &[Event]) -> bool {
    if sub.is_empty() { return true; }
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

#[test]
fn test_identify_hot_pixel_threshold() {
    let w = 8usize;
    let h = 6usize;

    let mut events: Vec<Event> = Vec::new();

    // High-frequency pixel (2,3): 10 events over ~1000 time units
    for i in 0..10 {
        events.push(Event { t_ns: (i as i64) * 100, x: 2, y: 3, p: 1, stream_id: 0 });
    }
    // Lower-frequency pixel (5,1): 3 events in the same span
    for i in 0..3 {
        events.push(Event { t_ns: 50 + (i as i64) * 250, x: 5, y: 1, p: 1, stream_id: 0 });
    }
    // Out-of-bounds events should be ignored by the kernel
    events.push(Event { t_ns: 999, x: 9999, y: 9999, p: 1, stream_id: 0 });

    // Duration ~ 999 (max 999 - min 0). Frequencies:
    // (2,3): 10/999 ~ 0.01001; (5,1): 3/999 ~ 0.003
    // Threshold picks only (2,3)
    let hot = identify_hot_pixel(&events, w, h, 0.009);
    assert_eq!(hot, vec![(2u16, 3u16)]);

    // Lower threshold picks both; result sorted lexicographically by (y, then x)
    // (5,1) has y=1, then (2,3) y=3
    let hot2 = identify_hot_pixel(&events, w, h, 0.002);
    assert_eq!(hot2, vec![(5u16, 1u16), (2u16, 3u16)]);
}

#[test]
fn test_drop_pixel_removes_coords_order_preserved() {
    let w = 10usize;
    let h = 10usize;

    let events = vec![
        Event { t_ns: 10, x: 0, y: 0, p: 1,  stream_id: 0 },
        Event { t_ns: 11, x: 2, y: 2, p: -1, stream_id: 0 },
        Event { t_ns: 12, x: 0, y: 0, p: 1,  stream_id: 0 },
        Event { t_ns: 13, x: 3, y: 1, p: 1,  stream_id: 0 },
    ];

    // Drop (0,0) and (3,1); include an OOB coord which should be ignored
    let coords = vec![(0u16, 0u16), (3u16, 1u16), (999u16, 999u16)];
    let kept = drop_pixel(&events, w, h, &coords);

    // Only the middle event at (2,2) should remain
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0], events[1]);

    // Order preservation (trivial here since length=1) and stable subsequence
    assert!(is_stable_subsequence(&kept, &events));
}

#[test]
fn test_edge_cases_empty_and_degenerate() {
    let w = 4usize;
    let h = 4usize;

    // identify_hot_pixel: empty input
    let empty: Vec<Event> = Vec::new();
    let hot_empty = identify_hot_pixel(&empty, w, h, 1.0);
    assert!(hot_empty.is_empty());

    // identify_hot_pixel: duration <= 0 (all timestamps equal) -> empty set
    let same_t = vec![
        Event { t_ns: 100, x: 1, y: 1, p: 1, stream_id: 0 },
        Event { t_ns: 100, x: 1, y: 1, p: 1, stream_id: 0 },
        Event { t_ns: 100, x: 2, y: 2, p: 1, stream_id: 0 },
    ];
    let hot_deg = identify_hot_pixel(&same_t, w, h, 0.0);
    assert!(hot_deg.is_empty(), "degenerate duration must yield no hot pixels");

    // drop_pixel: empty coords -> identity
    let events = vec![
        Event { t_ns: 0, x: 0, y: 0, p: 1, stream_id: 0 },
        Event { t_ns: 1, x: 1, y: 1, p: 1, stream_id: 0 },
        Event { t_ns: 2, x: 2, y: 2, p: 1, stream_id: 0 },
    ];
    let kept_identity = drop_pixel(&events, w, h, &[]);
    assert_eq!(kept_identity, events);
    assert!(is_stable_subsequence(&kept_identity, &events));

    // drop_pixel: empty events -> empty
    let kept_empty = drop_pixel(&[], w, h, &[(0u16, 0u16)]);
    assert!(kept_empty.is_empty());
}
