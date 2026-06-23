//! Usage example from Rust. Run with:
//!   cargo run --example basic -p pso-core

use pso_core::benchmarks::rastrigin;
use pso_core::prelude::*;

fn main() {
    let space = ContinuousSpace::new(vec![(-5.12, 5.12); 2]);
    let velocity = InertiaVelocity::new(0.9, 1.49445, 1.49445).with_decay(0.4);
    let params = PsoParams {
        n_particles: 40,
        max_iterations: 200,
        seed: Some(42),
        record_history: true,
        ..Default::default()
    };

    let pso = Pso::new(space, velocity, GlobalBest::new(), params);
    let result = pso.minimize(rastrigin);

    println!("Best position: {:?}", result.best_position);
    println!("Best value:    {:.6}", result.best_value);
    println!("Iterations:    {}", result.history.iterations());
}
