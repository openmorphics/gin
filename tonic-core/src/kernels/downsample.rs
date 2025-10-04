// Event downsampling kernels: integrator and differentiator modes
// Semantics:
// - Spatial grouping by block size `downsample_factor` (factor >= 1):
//   reduced_x = floor(x / factor), reduced_y = floor(y / factor)
// - Polarity contribution: p > 0 => +1, else => -1 (treat 0 and negative as OFF/-1)
// - Integrator mode: accumulate signed counts per reduced cell; when |accum| >= threshold,
//   emit a single event at the current input timestamp with p=1 for +, p=0 for - and reset
//   accumulator for that cell to 0.
// - Differentiator mode: track a per-cell thresholded state in {-1, 0, +1}. After each input
//   event update, compute new_state:
//        +1 if accum >= threshold
//        -1 if -accum >= threshold
//         0 otherwise
//   Emit an output event only for 0->(+1) or 0->(-1) transitions (positive edge in the
//   corresponding sign channel), then reset that cell accumulator to 0 to mirror an integrate-&-fire
//   spike that discharges. Transitions (+1)->0 or (-1)->0 do not emit events.
// - Determinism: preserves the input event order; at most one output event per input event.
// - Edge cases:
//   * Empty input => empty output
//   * Out-of-bounds x/y (>= sensor size) are ignored
//   * factor <= 0 is treated as 1
//   * threshold <= 0: every in-bounds input event produces a single output event mapped
//     to its reduced cell with the corresponding sign
//
// Note: Python reference (tonic/functional/event_downsampling.py) implements time-sliced
// accumulation via to_frame and emits events at dt-slice boundaries. This kernel emits at the
// exact input event timestamps while preserving integrate-and-fire and differentiator (edge)
// semantics over spatially downsampled neighborhoods. The Python adapter performs routing to
// ensure consistent behavior at the API level.

use crate::events::Event;

#[inline]
fn pol_to_signed(p: i8) -> i32 {
    if p > 0 { 1 } else { -1 }
}

#[inline]
fn reduced_dims(sensor_w: usize, sensor_h: usize, factor: usize) -> (usize, usize) {
    let f = factor.max(1);
    // ceil division for grid coverage; indices still computed by floor(x/f)
    let rw = (sensor_w + f - 1) / f;
    let rh = (sensor_h + f - 1) / f;
    (rw, rh)
}

pub fn event_integrator_downsample(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    downsample_factor: usize,
    noise_threshold: i32,
) -> Vec<Event> {
    let n = events.len();
    if n == 0 || sensor_w == 0 || sensor_h == 0 {
        return Vec::new();
    }
    let f = downsample_factor.max(1);
    let (rw, rh) = reduced_dims(sensor_w, sensor_h, f);
    let thr = noise_threshold;

    let mut accum = vec![0i32; rw.saturating_mul(rh)];
    let mut out = Vec::with_capacity(n.min(1024)); // heuristic

    // If threshold <= 0, pass-through mapped to reduced grid (emit per event)
    if thr <= 0 {
        for &ev in events {
            let x = ev.x as usize;
            let y = ev.y as usize;
            if x >= sensor_w || y >= sensor_h {
                continue;
            }
            let rx = x / f;
            let ry = y / f;
            let p_out: i8 = if pol_to_signed(ev.p) > 0 { 1 } else { 0 };
            out.push(Event {
                t_ns: ev.t_ns,
                x: rx as u16,
                y: ry as u16,
                p: p_out,
                stream_id: 0,
            });
        }
        return out;
    }

    for &ev in events {
        let x = ev.x as usize;
        let y = ev.y as usize;
        if x >= sensor_w || y >= sensor_h {
            continue; // ignore out-of-bounds safely
        }
        let rx = x / f;
        let ry = y / f;
        let idx = ry * rw + rx;

        // integrate signed polarity
        accum[idx] = accum[idx].saturating_add(pol_to_signed(ev.p));

        // fire if threshold crossed (positive wins ties if both were possible, but both cannot happen at once)
        if accum[idx] >= thr {
            out.push(Event {
                t_ns: ev.t_ns,
                x: rx as u16,
                y: ry as u16,
                p: 1, // positive spike
                stream_id: 0,
            });
            accum[idx] = 0; // reset after spike
        } else if -accum[idx] >= thr {
            out.push(Event {
                t_ns: ev.t_ns,
                x: rx as u16,
                y: ry as u16,
                p: 0, // negative spike
                stream_id: 0,
            });
            accum[idx] = 0; // reset after spike
        }
    }

    out
}

pub fn event_differentiator_downsample(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    downsample_factor: usize,
    noise_threshold: i32,
) -> Vec<Event> {
    let n = events.len();
    if n == 0 || sensor_w == 0 || sensor_h == 0 {
        return Vec::new();
    }
    let f = downsample_factor.max(1);
    let (rw, rh) = reduced_dims(sensor_w, sensor_h, f);
    let thr = noise_threshold;

    let mut accum = vec![0i32; rw.saturating_mul(rh)];
    // last_state per cell: -1, 0, +1
    let mut last_state = vec![0i8; rw.saturating_mul(rh)];
    let mut out = Vec::with_capacity(n.min(1024));

    // For threshold <= 0, emit per event like integrator (each event is a rising edge)
    if thr <= 0 {
        for &ev in events {
            let x = ev.x as usize;
            let y = ev.y as usize;
            if x >= sensor_w || y >= sensor_h { continue; }
            let rx = x / f;
            let ry = y / f;
            let p_out: i8 = if pol_to_signed(ev.p) > 0 { 1 } else { 0 };
            out.push(Event { t_ns: ev.t_ns, x: rx as u16, y: ry as u16, p: p_out, stream_id: 0 });
        }
        return out;
    }

    for &ev in events {
        let x = ev.x as usize;
        let y = ev.y as usize;
        if x >= sensor_w || y >= sensor_h {
            continue;
        }
        let rx = x / f;
        let ry = y / f;
        let idx = ry * rw + rx;

        accum[idx] = accum[idx].saturating_add(pol_to_signed(ev.p));

        let pos = accum[idx] >= thr;
        let neg = -accum[idx] >= thr;
        let new_state: i8 = if pos && !neg {
            1
        } else if neg && !pos {
            -1
        } else {
            0
        };

        // Positive-edge only on each sign; reset accumulator on spike to mimic I&F discharge
        if new_state == 1 && last_state[idx] != 1 {
            out.push(Event {
                t_ns: ev.t_ns,
                x: rx as u16,
                y: ry as u16,
                p: 1,
                stream_id: 0,
            });
            last_state[idx] = 1;
            accum[idx] = 0;
        } else if new_state == -1 && last_state[idx] != -1 {
            out.push(Event {
                t_ns: ev.t_ns,
                x: rx as u16,
                y: ry as u16,
                p: 0,
                stream_id: 0,
            });
            last_state[idx] = -1;
            accum[idx] = 0;
        } else if new_state == 0 {
            // drop below threshold: arm future rising-edge detection
            last_state[idx] = 0;
            // do not reset accum here; natural decay is handled by event sequence integrating both signs
        }
    }

    out
}

/// Unified entrypoint toggled by `differentiate`.
pub fn event_downsample_integrator(
    events: &[Event],
    sensor_size: (u16, u16),
    downsample_factor: u8,
    noise_threshold: u32,
    differentiate: bool,
) -> Vec<Event> {
    let (w, h) = (sensor_size.0 as usize, sensor_size.1 as usize);
    let f = downsample_factor as usize;
    let thr = noise_threshold as i32;
    if differentiate {
        event_differentiator_downsample(events, w, h, f, thr)
    } else {
        event_integrator_downsample(events, w, h, f, thr)
    }
}