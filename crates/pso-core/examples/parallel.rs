//! Demonstrates `minimize_parallel`: parallel objective evaluation pays off
//! when the objective is expensive. Run with:
//!
//!     cargo run --release --example parallel -p pso-core

use std::hint::black_box;
use std::time::Instant;

use pso_core::benchmarks::sphere;
use pso_core::prelude::*;

/// An artificially expensive objective: it computes `sphere` but burns extra
/// CPU per evaluation to mimic a costly fitness function (a simulation, a model
/// evaluation, etc.), which is where parallelism helps.
fn expensive(x: &[f64]) -> f64 {
    let mut burn = 0.0_f64;
    for _ in 0..20_000 {
        for &xi in x {
            burn += (xi.sin() * xi.cos()).abs();
        }
    }
    black_box(burn); // prevent the compiler from optimizing the loop away
    sphere(x)
}

fn main() {
    let make = || {
        let space = ContinuousSpace::new(vec![(-5.12, 5.12); 5]);
        let params = PsoParams {
            n_particles: 64,
            max_iterations: 60,
            seed: Some(42),
            record_history: false,
            ..Default::default()
        };
        Pso::new(space, InertiaVelocity::new(0.729, 1.49445, 1.49445), GlobalBest::new(), params)
    };

    let t0 = Instant::now();
    let seq = make().minimize(expensive);
    let seq_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let par = make().minimize_parallel(expensive);
    let par_ms = t1.elapsed().as_secs_f64() * 1000.0;

    println!("sequential : {seq_ms:8.1} ms   best = {:.3e}", seq.best_value);
    println!("parallel   : {par_ms:8.1} ms   best = {:.3e}", par.best_value);
    println!("speedup    : {:.2}x", seq_ms / par_ms);
    println!("(note: sequential uses asynchronous updates, parallel synchronous,");
    println!(" so the best values differ slightly — both are valid PSO schemes.)");
}
