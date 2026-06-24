//! Standard test functions for optimization, with their known global
//! optimum. They allow validating the optimizer and measuring the real error.
//!
//! Phase 1: Sphere, Rastrigin, Rosenbrock.
//! Phase 2: Ackley, Griewank, Schwefel.
//! CEC family: Bent Cigar, Discus, Elliptic, Zakharov, Levy, Expanded Schaffer.
//! Grey: grey_sphere (operates on grey numbers).

mod functions;
mod grey;

pub use functions::{
    ackley, bent_cigar, discus, elliptic, expanded_schaffer, griewank, levy, meta, rastrigin,
    rosenbrock, schwefel, sphere, zakharov, Benchmark, ALL,
};
pub use grey::{grey_meta, grey_sphere, GreyBenchmark, GREY_ALL, GREY_SPHERE};
