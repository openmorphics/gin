use tonic_core::kernels::denoise::denoise;
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

#[test]
fn denoise_filters_some_events_and_is_deterministic() {
    let w = 10usize;
    let h = 10usize;
    let filter_time: i64 = 100;

    let mut events: Vec<Event> = Vec::new();

    // Construct clustered neighbor activity so some events are kept.
    // Pairs on (5,5)->(6,5) with short temporal gap so the second sees a recent neighbor.
    // Spacing between pairs well below filter_time so first of the next pair also sees a recent neighbor.
    for k in 0..20 {
        let t0 = 10_000 + (k as i64) * 50; // >> filter_time so initial memories (100) are not > t0
        events.push(Event { t_ns: t0,     x: 5, y: 5, p: 1, stream_id: 0 });
        events.push(Event { t_ns: t0 + 10, x: 6, y: 5, p: 1, stream_id: 0 });
    }

    // Add isolated events far apart in time and space; these should be dropped
    // because no 4-neighbor has memory > t at their times.
    for i in 0..50 {
        let t = 1_000_000 + (i as i64) * (filter_time as i64 * 2);
        let x = (i % w) as u16;
        let y = ((i / w) % h) as u16;
        events.push(Event { t_ns: t, x, y, p: 1, stream_id: 0 });
    }

    // Shuffle-like interleave: add a slow diagonal walk, still far beyond filter_time
    for i in 0..30 {
        let t = 2_000_000 + (i as i64) * (filter_time as i64 * 3);
        let x = (i % w) as u16;
        let y = (i % h) as u16;
        events.push(Event { t_ns: t, x, y, p: 1, stream_id: 0 });
    }

    // Ensure temporal order since kernel expects sorted timestamps
    events.sort_by_key(|e| e.t_ns);

    let out1 = denoise(&events, w, h, filter_time);

    // Basic invariants similar to Python tests:
    assert!(!out1.is_empty(), "Not all events should be filtered");
    assert!(out1.len() < events.len(), "Result should be fewer events than original stream");
    assert!(
        is_stable_subsequence(&out1, &events),
        "Denoise must not invent events and must preserve original order"
    );

    // Determinism
    let out2 = denoise(&events, w, h, filter_time);
    assert_eq!(out1, out2, "Denoise must be deterministic for the same input");
}