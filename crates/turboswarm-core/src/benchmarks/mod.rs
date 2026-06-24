//! Standard test functions for optimization, with their known global
//! optimum. They allow validating the optimizer and measuring the real error.
//!
//! Phase 1: Sphere, Rastrigin, Rosenbrock.
//! Phase 2: Ackley, Griewank, Schwefel.
//! CEC family: Bent Cigar, Discus, Elliptic, Zakharov, Levy, Expanded Schaffer.

mod functions;

pub use functions::{
    ackley, bent_cigar, discus, elliptic, expanded_schaffer, griewank, levy, meta, rastrigin,
    rosenbrock, schwefel, sphere, zakharov, Benchmark, ALL,
};
