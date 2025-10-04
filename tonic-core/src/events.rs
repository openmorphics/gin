#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    /// Timestamp in nanoseconds
    pub t_ns: i64,
    /// X-coordinate (column)
    pub x: u16,
    /// Y-coordinate (row)
    pub y: u16,
    /// Polarity: expected values 0/1 or -1/1. Kernel maps 0 -> -1.0.
    pub p: i8,
    /// Stream identifier for multi-sensor or sharded streams
    pub stream_id: u16,
}


/// Ensure a deterministic strict ordering over events using a stable lexicographic order:
/// (t_ns, y, x, p). This provides a tie-break policy for equal timestamps and makes
/// downstream kernels reproducible when inputs originate from multiple sources.
///
/// Notes:
/// - This function is stable: events that are equal in (t_ns,y,x,p) keep their original order
///   (e.g., different stream_id values with identical keys remain in input order).
/// - The chosen order aligns with the policy documented in docs/parity.md.
pub fn ensure_strict_order(evs: &mut [Event]) {
    use core::cmp::Ordering;
    evs.sort_by(|a, b| {
        match a.t_ns.cmp(&b.t_ns) {
            Ordering::Equal => match a.y.cmp(&b.y) {
                Ordering::Equal => match a.x.cmp(&b.x) {
                    Ordering::Equal => a.p.cmp(&b.p),
                    other => other,
                },
                other => other,
            },
            other => other,
        }
    });
}

/// Check whether a slice is already strictly ordered by (t_ns, y, x, p).
#[inline]
pub fn is_strictly_ordered(evs: &[Event]) -> bool {
    if evs.len() <= 1 {
        return true;
    }
    for i in 1..evs.len() {
        let a = &evs[i - 1];
        let b = &evs[i];
        if a.t_ns > b.t_ns {
            return false;
        }
        if a.t_ns == b.t_ns {
            if a.y > b.y {
                return false;
            }
            if a.y == b.y {
                if a.x > b.x {
                    return false;
                }
                if a.x == b.x {
                    if a.p > b.p {
                        return false;
                    }
                }
            }
        }
    }
    true
}
