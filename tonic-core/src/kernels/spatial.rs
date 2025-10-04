use crate::events::Event;
use rand::thread_rng;
use rand_distr::StandardNormal;
use rand_distr::Distribution;

/// Compute a numerically safe Cholesky factorization of a 2x2 covariance matrix:
/// [[var_x, sigma_xy], [sigma_xy, var_y]]
/// Returns (L11, L21, L22) such that L = [[L11, 0], [L21, L22]] and L * L^T = cov.
/// Falls back to small diagonal when inputs are near-singular or invalid.
fn cholesky_2x2(var_x: f64, sigma_xy: f64, var_y: f64) -> (f64, f64, f64) {
    // Ensure non-negative variances and positive-definite-ish matrix
    let eps = 1e-12f64;
    let a = if var_x.is_finite() && var_x > 0.0 { var_x } else { eps };
    let c = if var_y.is_finite() && var_y > 0.0 { var_y } else { eps };
    let b = if sigma_xy.is_finite() { sigma_xy } else { 0.0 };

    let l11 = a.sqrt();
    // If a is too small, fall back to diagonal
    if !l11.is_finite() || l11 <= 0.0 {
        let d = eps.sqrt();
        return (d, 0.0, d);
    }
    let l21 = b / l11;
    let s22 = (c - l21 * l21).max(eps);
    let l22 = s22.sqrt();
    if !l22.is_finite() || l22 <= 0.0 {
        let d = eps.sqrt();
        (l11, 0.0, d)
    } else {
        (l11, l21, l22)
    }
}

/// Spatial jitter: add zero-mean Gaussian noise to (x,y) per event using a 2x2 covariance
/// [[var_x, sigma_xy],[sigma_xy, var_y]].
/// - If clip_outliers is true: drop events whose jittered (x,y) fall outside [0,W) x [0,H).
/// - If clip_outliers is false: clamp jittered (x,y) into the valid sensor bounds.
/// - Truncates toward zero when casting to integer coordinates (to mirror NumPy assignment into int fields).
///
/// Note: Python numpy implementation can store negative coordinates in int fields; in Rust we store u16,
/// so we clamp into bounds for the non-clipping case.
pub fn spatial_jitter(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    var_x: f64,
    var_y: f64,
    sigma_xy: f64,
    clip_outliers: bool,
) -> Vec<Event> {
    if events.is_empty() || sensor_w == 0 || sensor_h == 0 {
        return Vec::new();
    }

    let (l11, l21, l22) = cholesky_2x2(var_x.max(0.0), sigma_xy, var_y.max(0.0));
    let mut rng = thread_rng();

    let mut out = Vec::with_capacity(events.len());
    for ev in events {
        let z1: f64 = StandardNormal.sample(&mut rng);
        let z2: f64 = StandardNormal.sample(&mut rng);
        let dx = l11 * z1;
        let dy = l21 * z1 + l22 * z2;

        // Truncate toward zero like NumPy assignment into integer dtype
        let nx_i = (ev.x as f64 + dx).trunc() as i64;
        let ny_i = (ev.y as f64 + dy).trunc() as i64;

        if clip_outliers {
            if nx_i < 0
                || ny_i < 0
                || nx_i >= sensor_w as i64
                || ny_i >= sensor_h as i64
            {
                continue;
            }
            // Safe to cast
            let nx = nx_i as u16;
            let ny = ny_i as u16;
            out.push(Event {
                t_ns: ev.t_ns,
                x: nx,
                y: ny,
                p: ev.p,
                stream_id: ev.stream_id,
            });
        } else {
            // Clamp into bounds to respect u16 storage for pixel coordinates
            let nx_i = nx_i.clamp(0, sensor_w.saturating_sub(1) as i64);
            let ny_i = ny_i.clamp(0, sensor_h.saturating_sub(1) as i64);
            let nx = nx_i as u16;
            let ny = ny_i as u16;
            out.push(Event {
                t_ns: ev.t_ns,
                x: nx,
                y: ny,
                p: ev.p,
                stream_id: ev.stream_id,
            });
        }
    }
    out
}