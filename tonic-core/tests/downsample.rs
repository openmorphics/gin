// Tests for EventDownsampling kernels (integrator and differentiator)
use tonic_core::Event;
use tonic_core::kernels::downsample::{
    event_downsample_integrator,
    event_integrator_downsample,
    event_differentiator_downsample,
};

fn ev(t: i64, x: u16, y: u16, p: i8) -> Event {
    Event { t_ns: t, x, y, p, stream_id: 0 }
}

#[test]
fn test_empty_input_returns_empty() {
    let events: Vec<Event> = vec![];
    let out_int = event_downsample_integrator(&events, (10, 10), 1, 1, false);
    let out_diff = event_downsample_integrator(&events, (10, 10), 1, 1, true);
    assert!(out_int.is_empty(), "Integrator should return empty on empty input");
    assert!(out_diff.is_empty(), "Differentiator should return empty on empty input");
}

#[test]
fn test_spatial_neighborhood_behavior() {
    // 4x4 sensor, factor=2 => reduced grid 2x2
    let w = 4u16;
    let h = 4u16;
    let factor = 2u8;
    let thr = 2u32;

    // Place events in top-left quadrant (x,y in {0,1}) -> reduced cell (0,0)
    let mut evs = Vec::new();
    evs.push(ev(1, 0, 0, 1));
    evs.push(ev(2, 1, 1, 1));
    evs.push(ev(3, 1, 0, -1)); // cancels one
    // After 3 events: accum = 1, threshold=2 -> no spike yet
    let out_none = event_downsample_integrator(&evs, (w, h), factor, thr, false);
    assert!(out_none.is_empty(), "No spike expected before threshold is reached");

    // Add another positive event in same neighborhood to reach threshold
    evs.push(ev(4, 0, 1, 1));
    let out = event_downsample_integrator(&evs, (w, h), factor, thr, false);
    assert_eq!(out.len(), 1, "Single spike when threshold crossed");
    let o = &out[0];
    assert_eq!((o.x, o.y), (0, 0), "Reduced coordinates should be (0,0)");
    assert_eq!(o.p, 1, "Positive spike expected");
    assert_eq!(o.t_ns, 4, "Spike emitted at time of threshold-crossing event");
}

#[test]
fn test_integrator_accumulates_until_threshold() {
    // 2x2 sensor, factor=2 => reduced to 1x1 cell
    let sensor = (2u16, 2u16);
    let thr = 3u32;
    let evs = vec![
        ev(10, 0, 0, 1),
        ev(11, 1, 0, 1),
        ev(12, 0, 1, 1),
    ];
    let out = event_downsample_integrator(&evs, sensor, 2, thr, false);
    assert_eq!(out.len(), 1, "Exactly one spike at threshold crossing");
    assert_eq!(out[0].x, 0);
    assert_eq!(out[0].y, 0);
    assert_eq!(out[0].p, 1);
    assert_eq!(out[0].t_ns, 12);
}

#[test]
fn test_differentiator_outputs_significant_changes() {
    // Single reduced cell; threshold=2; we expect rising-edge spikes for + and - separately
    let sensor = (2u16, 2u16);
    let thr = 2i32;
    let evs = vec![
        ev(1, 0, 0, 1),   // accum=+1 -> no spike
        ev(2, 1, 0, 1),   // accum=+2 -> + spike at t=2 (reset, state=+1 then armed)
        ev(3, 0, 1, -1),  // accum=-1 -> state returns to 0 (no spike)
        ev(4, 1, 1, -1),  // accum=-2 -> - spike at t=4 (reset)
        ev(5, 0, 0, -1),  // accum=-1 -> no spike
        ev(6, 1, 0, 1),   // accum=0 -> state=0 (no spike)
        ev(7, 0, 1, 1),   // accum=+1 -> no spike
        ev(8, 1, 1, 1),   // accum=+2 -> + spike at t=8
    ];
    // Call the dedicated differentiator kernel to isolate semantics
    let out = event_differentiator_downsample(&evs, sensor.0 as usize, sensor.1 as usize, 2usize, thr);
    assert_eq!(out.len(), 3, "Expect + at t=2, - at t=4, + at t=8");
    assert_eq!(out[0].t_ns, 2);
    assert_eq!(out[0].p, 1);
    assert_eq!(out[1].t_ns, 4);
    assert_eq!(out[1].p, 0);
    assert_eq!(out[2].t_ns, 8);
    assert_eq!(out[2].p, 1);
}

#[test]
fn test_noise_threshold_filtering() {
    // threshold == 0 -> each in-bounds input produces an output mapped to reduced cell
    let sensor = (4u16, 2u16);
    let factor = 2u8;
    let thr0 = 0u32;
    let evs = vec![
        ev(1, 0, 0, 1),   // -> (0,0)
        ev(2, 1, 0, -1),  // -> (0,0)
        ev(3, 2, 1, 1),   // -> (1,1)
        ev(4, 3, 1, 1),   // -> (1,1)
        ev(5, 9, 9, 1),   // OOB -> dropped
    ];
    let out_int = event_downsample_integrator(&evs, sensor, factor, thr0, false);
    assert_eq!(out_int.len(), 4, "All in-bounds events should pass through when threshold==0");

    // Spot-check mapping
    assert_eq!((out_int[0].x, out_int[0].y), (0, 0));
    assert_eq!((out_int[2].x, out_int[2].y), (1, 1));
}

#[test]
fn test_deterministic_ordering() {
    // Ensure output preserves timestamp order and is deterministic
    let sensor = (2u16, 2u16);
    let thr = 2u32;
    let evs = vec![
        ev(100, 0, 0, 1),
        ev(100, 1, 0, 1), // same timestamp, threshold crossed -> spike at this event
        ev(101, 0, 1, -1),
        ev(101, 1, 1, -1), // another threshold crossing
    ];
    let out = event_downsample_integrator(&evs, sensor, 2, thr, false);
    assert_eq!(out.len(), 2);
    assert!(out[0].t_ns <= out[1].t_ns, "Timestamps must be non-decreasing");
    assert_eq!(out[0].t_ns, 100);
    assert_eq!(out[1].t_ns, 101);
}

#[test]
fn test_out_of_bounds_handling() {
    // OOB events must be ignored (not counted towards accumulators)
    let sensor = (3u16, 3u16);
    let thr = 2u32;
    let evs = vec![
        ev(1,  3, 0, 1),  // OOB x
        ev(2,  0, 3, 1),  // OOB y
        ev(3,  3, 3, 1),  // OOB both
        ev(4,  1, 1, 1),  // in-bounds
        ev(5,  1, 1, 1),  // in-bounds -> threshold reached here
    ];
    let out = event_downsample_integrator(&evs, sensor, 1, thr, false);
    assert_eq!(out.len(), 1, "Only in-bounds events should contribute to spikes");
    assert_eq!(out[0].t_ns, 5);
    assert_eq!((out[0].x, out[0].y), (1, 1));
}

#[test]
fn test_integrator_internal_function_equivalence() {
    // Cross-check unified entrypoint vs direct integrator helper for identical inputs
    let evs = vec![
        ev(1, 0, 0, 1),
        ev(2, 0, 0, 1),
        ev(3, 0, 0, -1),
        ev(4, 0, 0, 1),
    ];
    let out_unified = event_downsample_integrator(&evs, (4, 4), 2, 2, false);
    let out_direct = event_integrator_downsample(&evs, 4, 4, 2, 2);
    assert_eq!(out_unified, out_direct, "Unified entrypoint and direct integrator must match");
}