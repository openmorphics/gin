use tonic_core::kernels::decimate::decimate;
use tonic_core::Event;

fn make_events_single_pixel(n: usize, x: u16, y: u16, start_t: i64) -> Vec<Event> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(Event {
            t_ns: start_t + i as i64,
            x,
            y,
            p: 1,
            stream_id: 0,
        });
    }
    out
}

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
fn single_pixel_every_tenth() {
    let w = 1usize;
    let h = 1usize;
    let n = 10usize;

    let events = make_events_single_pixel(1000, 0, 0, 0);
    let kept = decimate(&events, w, h, n);

    // Expect exactly 100 events kept (every 10th)
    assert_eq!(kept.len(), 100);

    // Stable subset property and determinism
    assert!(is_stable_subsequence(&kept, &events));
    let kept2 = decimate(&events, w, h, n);
    assert_eq!(kept, kept2);
}

#[test]
fn idempotence_of_composition() {
    let w = 1usize;
    let h = 1usize;
    let n1 = 5usize;
    let n2 = 4usize;

    let events = make_events_single_pixel(1000, 0, 0, 0);

    let first = decimate(&events, w, h, n1);
    let composed = decimate(&first, w, h, n2);

    let direct = decimate(&events, w, h, n1 * n2);

    assert_eq!(
        composed, direct,
        "decimate(n1) then decimate(n2) must equal decimate(n1*n2)"
    );

    // Order preserved and subset of original
    assert!(is_stable_subsequence(&composed, &events));
}

#[test]
fn multi_pixel_preserves_per_pixel_rate_and_order() {
    let w = 4usize;
    let h = 3usize;
    let n = 3usize;

    // Interleave events across multiple pixels
    let mut events: Vec<Event> = Vec::new();
    for i in 0..300 {
        let x = (i % w) as u16;
        let y = ((i / w) % h) as u16;
        events.push(Event { t_ns: i as i64, x, y, p: 1, stream_id: 0 });
    }

    let kept = decimate(&events, w, h, n);

    // Determinism
    let kept2 = decimate(&events, w, h, n);
    assert_eq!(kept, kept2);

    // Stable subsequence (original order preserved)
    assert!(is_stable_subsequence(&kept, &events));

    // Sanity: kept is smaller but non-zero
    assert!(!kept.is_empty());
    assert!(kept.len() < events.len());
}