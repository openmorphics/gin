use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tonic_core::kernels::frame::{
    to_frame_event_count, to_frame_n_time_bins, to_frame_time_window,
};
use tonic_core::Event;

const DURATION: i64 = 1_000_000;
const N_POLARITIES: usize = 2;

fn make_events(n: usize, w: usize, h: usize, seed: u64) -> Vec<Event> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut events = Vec::with_capacity(n);
    if n == 0 {
        return events;
    }
    for i in 0..n {
        let t = if n > 1 {
            ((i as i64) * DURATION) / ((n - 1) as i64)
        } else {
            0
        };
        let x = rng.gen_range(0..w) as u16;
        let y = rng.gen_range(0..h) as u16;
        let p = if rng.gen_bool(0.5) { 1i8 } else { 0i8 };
        events.push(Event { t, x, y, p });
    }
    events
}

pub fn frame_time_window_bench(c: &mut Criterion) {
    // Scenarios per spec:
    // - time_window mode: time_window=30_000, overlap=0; include_incomplete=false
    let sizes = [(128usize, 128usize), (240, 180)];
    let counts = [10_000usize, 50_000usize];

    let mut group = c.benchmark_group("frame/time_window");

    for &(w, h) in &sizes {
        for &n in &counts {
            let seed = 0xCAFEBABE ^ ((w as u64) << 32) ^ ((h as u64) << 16) ^ (n as u64) ^ 1;
            let events = make_events(n, w, h, seed);
            group.throughput(Throughput::Elements(n as u64));
            let id = BenchmarkId::new(format!("{}x{}-n{}", w, h, n), "tw=30000");
            group.bench_with_input(id, &(events.as_slice(), w, h), |b, (evs, w_, h_)| {
                b.iter(|| {
                    let out = to_frame_time_window(
                        black_box(evs),
                        *w_,
                        *h_,
                        N_POLARITIES,
                        30_000,
                        0,
                        false,
                        None,
                        None,
                    )
                    .expect("kernel ok");
                    black_box(out);
                });
            });
        }
    }

    group.finish();
}

pub fn frame_event_count_bench(c: &mut Criterion) {
    // Scenarios per spec:
    // - event_count mode: event_count=1_000, overlap=0
    let sizes = [(128usize, 128usize), (240, 180)];
    let counts = [10_000usize, 50_000usize];

    let mut group = c.benchmark_group("frame/event_count");

    for &(w, h) in &sizes {
        for &n in &counts {
            let seed = 0xCAFEBABE ^ ((w as u64) << 32) ^ ((h as u64) << 16) ^ (n as u64) ^ 2;
            let events = make_events(n, w, h, seed);
            group.throughput(Throughput::Elements(n as u64));
            let id = BenchmarkId::new(format!("{}x{}-n{}", w, h, n), "ec=1000");
            group.bench_with_input(id, &(events.as_slice(), w, h), |b, (evs, w_, h_)| {
                b.iter(|| {
                    let out = to_frame_event_count(
                        black_box(evs),
                        *w_,
                        *h_,
                        N_POLARITIES,
                        1_000,
                        0,
                        false,
                    )
                    .expect("kernel ok");
                    black_box(out);
                });
            });
        }
    }

    group.finish();
}

pub fn frame_n_time_bins_bench(c: &mut Criterion) {
    // Scenarios per spec:
    // - n_time_bins mode: n_time_bins=10, overlap_frac=0.0
    let sizes = [(128usize, 128usize), (240, 180)];
    let counts = [10_000usize, 50_000usize];

    let mut group = c.benchmark_group("frame/n_time_bins");

    for &(w, h) in &sizes {
        for &n in &counts {
            let seed = 0xCAFEBABE ^ ((w as u64) << 32) ^ ((h as u64) << 16) ^ (n as u64) ^ 3;
            let events = make_events(n, w, h, seed);
            group.throughput(Throughput::Elements(n as u64));
            let id = BenchmarkId::new(format!("{}x{}-n{}", w, h, n), "T=10");
            group.bench_with_input(id, &(events.as_slice(), w, h), |b, (evs, w_, h_)| {
                b.iter(|| {
                    let out =
                        to_frame_n_time_bins(black_box(evs), *w_, *h_, N_POLARITIES, 10, 0.0)
                            .expect("kernel ok");
                    black_box(out);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    frame_time_window_bench,
    frame_event_count_bench,
    frame_n_time_bins_bench
);
criterion_main!(benches);