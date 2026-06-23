//! Standard test functions for optimization, with their known global
//! optimum. They allow validating the optimizer and measuring the real error.
//!
//! Phase 1: Sphere, Rastrigin, Rosenbrock.
//! Phase 2: Ackley, Griewank, Schwefel.

mod functions;

pub use functions::{
    ackley, griewank, meta, rastrigin, rosenbrock, schwefel, sphere, Benchmark, ALL,
};
