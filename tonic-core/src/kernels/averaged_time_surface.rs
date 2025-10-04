use crate::events::Event;

/// Decay function kind for HATS averaged time surfaces.
#[derive(Clone, Copy, Debug)]
pub enum DecayKind {
    /// Exponential decay: exp(-(t_i - t_j) / tau)
    Exp,
    /// Linear decay: -(t_i - t_j) / (3 * tau) + 1
    Lin,
}

#[inline]
fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Compute averaged time surfaces per (cell, polarity) using HATS semantics.
/// Returns a contiguous f32 buffer laid out as (C, P, S, S) in row-major order:
/// - C = number of cells = ceil(W / cell_size) * ceil(H / cell_size)
/// - P = number of polarities
/// - S = surface_size (must be odd)
///
/// Semantics (matching Python reference to_averaged_timesurface_numpy):
/// - Partition the sensor into a uniform grid of square cells of side `cell_size`.
/// - For each cell and polarity channel independently:
///   - Iterate events in input order, treating earlier elements as "previous" events.
///   - For each event i, build a time-surface centered at the event:
///       * Consider previous events j in the same cell and polarity where:
///           |x_j - x_i| ≤ rho, |y_j - y_i| ≤ rho, and t_j ≥ max(0, t_i - time_window)
///         Place a contribution at offset (dy = y_j - y_i, dx = x_j - x_i):
///             value = exp(-(t_i - t_j)/tau)   if decay == Exp
///                   = -(t_i - t_j)/(3*tau)+1 if decay == Lin
///       * Add +1 at the center (rho, rho) for the current event.
///   - Sum these per-event surfaces across all events in the (cell, polarity),
///     then divide element-wise by max(1, n_events) (i.e., by n_events if nonzero).
///
/// Determinism:
/// - Input order is preserved and defines the "previous" relation (no sorting).
/// - Out-of-bounds events (x ≥ W or y ≥ H) are ignored safely.
/// - Negative polarity indices are mapped to 0 (channel 0) when P > 1; if P == 1 everything maps to channel 0.
/// - Polarity indices ≥ P are skipped.
///
/// Edge cases:
/// - If W==0 or H==0 or P==0, returns an empty buffer.
/// - cell_size must be > 0.
/// - surface_size must be odd and ≤ cell_size.
/// - time_window and tau are used as f32 as in the Python reference; tau == 0.0 yields IEEE behavior.
pub fn to_averaged_time_surface(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
    cell_size: usize,
    surface_size: usize,
    time_window: f32,
    tau: f32,
    decay: DecayKind,
) -> Result<Vec<f32>, String> {
    if sensor_w == 0 || sensor_h == 0 || n_polarities == 0 {
        return Ok(Vec::new());
    }
    if cell_size == 0 {
        return Err("cell_size must be greater than zero.".to_string());
    }
    if surface_size == 0 {
        return Err("surface_size must be greater than zero.".to_string());
    }
    if surface_size % 2 == 0 {
        return Err("surface_size must be odd.".to_string());
    }
    if surface_size > cell_size {
        return Err("surface_size must be less than or equal to cell_size.".to_string());
    }

    let wgrid = ceil_div(sensor_w, cell_size);
    let hgrid = ceil_div(sensor_h, cell_size);
    let n_cells = wgrid.saturating_mul(hgrid);
    let s = surface_size;
    let rho = s / 2;

    // Output buffer (C, P, S, S)
    let mut hist = vec![0f32; n_cells * n_polarities * s * s];
    if n_cells == 0 {
        return Ok(hist);
    }

    // Local memories: for each cell and polarity, keep events in input order.
    // Store minimal fields to reduce memory footprint: (x, y, t)
    let mut locmems: Vec<Vec<Vec<(u16, u16, i64)>>> = vec![vec![Vec::new(); n_polarities]; n_cells];

    for ev in events.iter() {
        let x = ev.x as usize;
        let y = ev.y as usize;
        if x >= sensor_w || y >= sensor_h {
            continue;
        }
        let cx = x / cell_size;
        let cy = y / cell_size;
        if cx >= wgrid || cy >= hgrid {
            continue;
        }
        let cell_idx = cy * wgrid + cx;

        // Polarity channel mapping:
        // - If P==1, map everything to channel 0.
        // - Else, negative p maps to 0; p ≥ P is skipped.
        let p_idx = if n_polarities == 1 {
            0usize
        } else {
            if ev.p < 0 {
                0usize
            } else {
                let pu = ev.p as usize;
                if pu >= n_polarities {
                    continue;
                }
                pu
            }
        };

        locmems[cell_idx][p_idx].push((ev.x, ev.y, ev.t_ns));
    }

    // Helper to flatten (C,P,S,S) indexing
    #[inline]
    fn base_offset(cell: usize, p: usize, s: usize, n_p: usize) -> usize {
        ((cell * n_p) + p) * s * s
    }

    for c in 0..n_cells {
        for p in 0..n_polarities {
            let evs = &locmems[c][p];
            let n = evs.len();
            if n == 0 {
                continue;
            }
            let base = base_offset(c, p, s, n_polarities);

            for i in 0..n {
                let (xi, yi, ti) = evs[i];
                let ti_f = ti as f32;
                let t_start = (ti_f - time_window).max(0.0);

                // Accumulate contributions from previous events in window and neighborhood
                for j in 0..i {
                    let (xj, yj, tj) = evs[j];
                    if (tj as f32) < t_start {
                        continue;
                    }

                    let dx = (xj as i32) - (xi as i32);
                    let dy = (yj as i32) - (yi as i32);
                    if (dx.unsigned_abs() as usize) > rho || (dy.unsigned_abs() as usize) > rho {
                        continue;
                    }

                    let dt = ti_f - (tj as f32);
                    let val = match decay {
                        DecayKind::Exp => (-dt / tau).exp(),
                        DecayKind::Lin => (-(dt) / (3.0 * tau) + 1.0),
                    };

                    // Map (dy, dx) to surface coordinates (sy, sx) in [0, S)
                    let sy = (dy + (rho as i32)) as usize;
                    let sx = (dx + (rho as i32)) as usize;
                    let idx = base + sy * s + sx;
                    // Safe due to mask check
                    hist[idx] += val;
                }

                // Add the current event at center
                let center = base + rho * s + rho;
                hist[center] += 1.0;
            }

            // Average over number of events in this (cell, polarity)
            let denom = n as f32;
            if denom > 0.0 {
                for k in 0..(s * s) {
                    hist[base + k] /= denom;
                }
            }
        }
    }

    Ok(hist)
}