use crate::events::Event;
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;
use std::collections::HashSet;

/// Randomly drops events with probability `drop_probability` (0.0..=1.0).
/// Matches Python behavior by dropping exactly round(p * n) uniformly at random (without replacement).
/// Stable order of the kept events is preserved.
pub fn drop_event(events: &[Event], drop_probability: f64) -> Vec<Event> {
    let n = events.len();
    if n == 0 { return Vec::new(); }
    if drop_probability <= 0.0 { return events.to_vec(); }
    if drop_probability >= 1.0 { return Vec::new(); }
    let mut rng = thread_rng();
    let n_dropped = ((drop_probability * n as f64) + 0.5).floor() as usize;
    let n_dropped = n_dropped.min(n);
    // sample indices to drop by shuffling
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut rng);
    let drop_set: HashSet<usize> = idx.into_iter().take(n_dropped).collect();
    let mut out = Vec::with_capacity(n - n_dropped);
    for (i, ev) in events.iter().enumerate() {
        if !drop_set.contains(&i) {
            out.push(*ev);
        }
    }
    out
}

/// Drops events within a randomly chosen time interval of duration_ratio in [0,1).
/// The interval is sampled within [0, max_t] like the Python reference.
pub fn drop_by_time(events: &[Event], duration_ratio: f64) -> Vec<Event> {
    let n = events.len();
    if n == 0 { return Vec::new(); }
    if duration_ratio <= 0.0 { return events.to_vec(); }
    if duration_ratio >= 1.0 { return Vec::new(); }
    let max_t = events.iter().map(|e| e.t_ns).max().unwrap_or(0);
    let max_t_f = max_t as f64;
    let drop_dur = (max_t_f * duration_ratio).max(0.0);
    let mut rng = thread_rng();
    let low = 0.0;
    let high = (max_t_f - drop_dur).max(0.0);
    let start = if high > low {
        rng.gen_range(low..high)
    } else { low };
    let end = start + drop_dur;
    let mut out = Vec::with_capacity(n);
    for ev in events.iter() {
        let t = ev.t_ns as f64;
        if !(t >= start && t <= end) {
            out.push(*ev);
        }
    }
    out
}

/// Drops events located inside a randomly chosen axis-aligned box whose size is
/// area_ratio of the sensor resolution (approx via independent width/height scaling).
/// The selected area is inclusive on bounds (x<=x2, y<=y2) matching the NumPy reference.
pub fn drop_by_area(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    area_ratio: f64,
) -> Vec<Event> {
    let n = events.len();
    if n == 0 { return Vec::new(); }
    if area_ratio <= 0.0 { return events.to_vec(); }
    let ar = area_ratio.min(1.0).max(0.0);
    let mut cut_w = (sensor_w as f64 * ar) as usize;
    let mut cut_h = (sensor_h as f64 * ar) as usize;
    cut_w = cut_w.min(sensor_w);
    cut_h = cut_h.min(sensor_h);
    if cut_w == 0 || cut_h == 0 { return events.to_vec(); }
    // If box covers whole sensor, drop everything.
    if cut_w >= sensor_w && cut_h >= sensor_h {
        return Vec::new();
    }
    // pick top-left
    let max_x0 = sensor_w.saturating_sub(cut_w);
    let max_y0 = sensor_h.saturating_sub(cut_h);
    let mut rng = thread_rng();
    let x0 = if max_x0 > 0 { rng.gen_range(0..max_x0) } else { 0 };
    let y0 = if max_y0 > 0 { rng.gen_range(0..max_y0) } else { 0 };
    let x1 = x0 + cut_w - 1;
    let y1 = y0 + cut_h - 1;
    let mut out = Vec::with_capacity(n);
    for ev in events.iter() {
        let x = ev.x as usize;
        let y = ev.y as usize;
        let inside = x >= x0 && y >= y0 && x <= x1 && y <= y1;
        if !inside {
            out.push(*ev);
        }
    }
    out
}

/// Identify hot pixels by frequency threshold over the recording duration.
/// Frequency is defined as count(x,y) / (max_t - min_t) using the input timestamp units.
/// Returns unique (x,y) coordinates sorted lexicographically by (y, then x).
pub fn identify_hot_pixel(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    hot_pixel_frequency: f64,
) -> Vec<(u16, u16)> {
    let n = events.len();
    if n == 0 || sensor_w == 0 || sensor_h == 0 {
        return Vec::new();
    }

    let mut min_t = i64::MAX;
    let mut max_t = i64::MIN;
    for ev in events {
        if ev.t_ns < min_t { min_t = ev.t_ns; }
        if ev.t_ns > max_t { max_t = ev.t_ns; }
    }
    let duration = max_t.saturating_sub(min_t);
    if duration <= 0 {
        return Vec::new();
    }

    let mut counts = vec![0u32; sensor_w.saturating_mul(sensor_h)];
    for ev in events {
        let x = ev.x as usize;
        let y = ev.y as usize;
        if x < sensor_w && y < sensor_h {
            let idx = y * sensor_w + x;
            counts[idx] = counts[idx].saturating_add(1);
        }
    }

    let dur_f = duration as f64;
    let mut coords: Vec<(u16, u16)> = Vec::new();
    for y in 0..sensor_h {
        for x in 0..sensor_w {
            let idx = y * sensor_w + x;
            let freq = counts[idx] as f64 / dur_f;
            if freq >= hot_pixel_frequency {
                coords.push((x as u16, y as u16));
            }
        }
    }

    // sort by (y, then x) for deterministic ordering
    coords.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    coords
}

/// Drop any events whose (x,y) lies in the provided coordinate set.
/// Out-of-bounds events are kept. Original event order is preserved.
pub fn drop_pixel(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    coords: &[(u16, u16)],
) -> Vec<Event> {
    let n = events.len();
    if n == 0 {
        return Vec::new();
    }
    if coords.is_empty() {
        return events.to_vec();
    }

    let mut mask = std::collections::HashSet::<usize>::with_capacity(coords.len());
    for &(x, y) in coords {
        let xu = x as usize;
        let yu = y as usize;
        if xu < sensor_w && yu < sensor_h {
            mask.insert(yu * sensor_w + xu);
        }
    }
    if mask.is_empty() {
        return events.to_vec();
    }

    let mut out = Vec::with_capacity(n);
    for &ev in events.iter() {
        let x = ev.x as usize;
        let y = ev.y as usize;
        if x < sensor_w && y < sensor_h {
            let idx = y * sensor_w + x;
            if mask.contains(&idx) {
                continue; // drop
            }
        }
        out.push(ev);
    }
    out
}