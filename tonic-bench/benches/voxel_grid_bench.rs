use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tonic_core::kernels::voxel_grid::to_voxel_grid;
use tonic_core::Event;

const DURATION: i64 = 1_000_000;

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

pub fn voxel_grid_bench(c: &mut Criterion) {
    let sizes = [(128usize, 128usize), (240, 180), (346, 260)];
    let counts = [5_000usize, 10_000, 50_000];
    let time_bins = [10usize, 20usize];

    let mut group = c.benchmark_group("voxel_grid");

    for &(w, h) in &sizes {
        for &n in &counts {
            // Pre-generate deterministic events
            let seed = 0xC0FFEE ^ ((w as u64) << 32) ^ ((h as u64) << 16) ^ (n as u64);
            let events = make_events(n, w, h, seed);
            for &tb in &time_bins {
                group.throughput(Throughput::Elements(n as u64));
                let id = BenchmarkId::new(format!("{}x{}-n{}-T{}", w, h, n, tb), "");
                group.bench_with_input(id, &(events.as_slice(), w, h, tb), |b, (evs, w_, h_, tb_)| {
                    b.iter(|| {
                        // End-to-end kernel call
                        let out = to_voxel_grid(black_box(evs), *w_, *h_, *tb_);
                        black_box(out);
                    });
                });
            }
        }
    }

    group.finish();
}

criterion_group!(benches, voxel_grid_bench);
criterion_main!(benches);