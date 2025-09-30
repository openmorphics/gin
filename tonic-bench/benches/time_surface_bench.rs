use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tonic_core::kernels::time_surface::to_time_surface;
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

pub fn time_surface_bench(c: &mut Criterion) {
    // Scenarios:
    // - dt in {10_000, 30_000}, tau=30_000, overlap=0, include_incomplete=false
    // - Sensor sizes: (128,128,2), (240,180,2), (346,260,2)
    // - Event counts: 5k, 10k, 50k
    let sizes = [(128usize, 128usize), (240, 180), (346, 260)];
    let counts = [5_000usize, 10_000, 50_000];
    let dts = [10_000i64, 30_000i64];
    let tau = 30_000f64;

    let mut group = c.benchmark_group("time_surface");

    for &(w, h) in &sizes {
        for &n in &counts {
            // Pre-generate deterministic events
            let seed = 0xDEADBEEF ^ ((w as u64) << 32) ^ ((h as u64) << 16) ^ (n as u64);
            let events = make_events(n, w, h, seed);

            for &dt in &dts {
                group.throughput(Throughput::Elements(n as u64));
                let id = BenchmarkId::new(format!("{}x{}-n{}-dt{}", w, h, n, dt), "");
                group.bench_with_input(id, &(events.as_slice(), w, h, dt), |b, (evs, w_, h_, dt_)| {
                    b.iter(|| {
                        // End-to-end kernel call per iteration
                        let out = to_time_surface(
                            black_box(evs),
                            *w_,
                            *h_,
                            N_POLARITIES,
                            *dt_,
                            tau,
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
    }

    group.finish();
}

criterion_group!(benches, time_surface_bench);
criterion_main!(benches);