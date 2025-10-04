use crate::events::Event;

#[inline]
pub fn crop_box(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> Vec<Event> {
    // Early exits
    if events.is_empty() || sensor_w == 0 || sensor_h == 0 || w == 0 || h == 0 {
        return Vec::new();
    }
    // If origin is out of bounds, nothing to return.
    if x0 >= sensor_w || y0 >= sensor_h {
        return Vec::new();
    }

    // Clamp target dimensions to remain within sensor bounds.
    let max_w = sensor_w - x0;
    let max_h = sensor_h - y0;
    let tw = w.min(max_w);
    let th = h.min(max_h);
    if tw == 0 || th == 0 {
        return Vec::new();
    }

    let x1 = x0 + tw;
    let y1 = y0 + th;

    // Keep events with x in [x0, x1) and y in [y0, y1). Guard against OOB events.
    let mut out = Vec::with_capacity(events.len());
    for &ev in events.iter() {
        let x = ev.x as usize;
        let y = ev.y as usize;
        if x < sensor_w && y < sensor_h && x >= x0 && x < x1 && y >= y0 && y < y1 {
            let nx = (x - x0) as u16;
            let ny = (y - y0) as u16;
            out.push(Event { t_ns: ev.t_ns, x: nx, y: ny, p: ev.p, stream_id: ev.stream_id });
        }
    }
    out
}

pub fn center_crop(
    events: &[Event],
    sensor_w: usize,
    sensor_h: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<Event> {
    if events.is_empty() || sensor_w == 0 || sensor_h == 0 || target_w == 0 || target_h == 0 {
        return Vec::new();
    }
    // Clamp target to sensor dims
    let tw = target_w.min(sensor_w);
    let th = target_h.min(sensor_h);

    if tw == 0 || th == 0 {
        return Vec::new();
    }

    // Center offsets using floor division
    let ox = (sensor_w - tw) / 2;
    let oy = (sensor_h - th) / 2;

    crop_box(events, sensor_w, sensor_h, ox, oy, tw, th)
}