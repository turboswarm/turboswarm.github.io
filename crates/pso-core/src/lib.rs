//! # pso-core
//!
//! Particle Swarm Optimization (PSO) core: extensible and modular.
//! Pure Rust computation, with no FFI dependencies; the Python bindings live
//! in the separate `pso-py` crate.
//!
//! ## Core idea
//!
//! The PSO loop ([`Pso`](pso::Pso)) knows nothing about any concrete variant.
//! What changes between variants lives behind three traits
//! ([`SearchSpace`](traits::SearchSpace), [`Velocity`](traits::Velocity),
//! [`Topology`](traits::Topology)). Adding a variant = implementing a trait,
//! without touching the core. Internally, all positions and velocities are
//! `Vec<f64>`; the integer/real difference lives solely in
//! [`SearchSpace::decode`](traits::SearchSpace::decode).
//!
//! ## What it includes
//!
//! - **Spaces** ([`spaces`]): [`ContinuousSpace`](spaces::ContinuousSpace)
//!   (real) and [`IntegerSpace`](spaces::IntegerSpace) (integer, with
//!   configurable [`Discretization`](spaces::Discretization)).
//! - **Velocity variants** ([`velocity`]):
//!   [`InertiaVelocity`](velocity::InertiaVelocity) (Shi-Eberhart),
//!   [`ConstrictionVelocity`](velocity::ConstrictionVelocity) (Clerc-Kennedy) and
//!   [`FipsVelocity`](velocity::FipsVelocity) (Fully Informed PSO).
//! - **Topologies** ([`topology`]): [`GlobalBest`](topology::GlobalBest),
//!   [`Ring`](topology::Ring) (lbest ring), [`VonNeumann`](topology::VonNeumann)
//!   (2D grid) and [`Random`](topology::Random).
//! - **Benchmarks** ([`benchmarks`]): sphere, rastrigin, rosenbrock, ackley,
//!   griewank and schwefel, with their metadata ([`benchmarks::Benchmark`]).
//! - **History** ([`History`](history::History)): a complete trace of the swarm
//!   for visualization.
//!
//! ## Minimal example
//!
//! ```
//! use pso_core::prelude::*;
//!
//! let space = ContinuousSpace::uniform(2, -5.12, 5.12);
//! let velocity = InertiaVelocity::new(0.729, 1.49445, 1.49445);
//! let params = PsoParams { seed: Some(42), ..Default::default() };
//!
//! let pso = Pso::new(space, velocity, GlobalBest::new(), params);
//! let result = pso.minimize(|x| pso_core::benchmarks::sphere(x));
//!
//! assert!(result.best_value < 1e-3);
//! ```
//!
//! ## How to extend
//!
//! For a new variant, implement [`Velocity`](traits::Velocity)
//! (template: `velocity/inertia.rs`); for a new topology, implement
//! [`Topology`](traits::Topology) by defining
//! [`Topology::neighbors`](traits::Topology::neighbors). See the `README.md`
//! for the full guide.
#![warn(missing_docs)]

pub mod benchmarks;
pub mod history;
pub mod mopso;
pub mod params;
pub mod pso;
pub mod spaces;
pub mod swarm;
pub mod topology;
pub mod traits;
pub mod velocity;

/// Convenient re-exports for common usage.
pub mod prelude {
    pub use crate::history::History;
    pub use crate::mopso::{hypervolume, MoSolution, Mopso, MopsoParams, MopsoResult};
    pub use crate::params::PsoParams;
    pub use crate::pso::{IterationInfo, Pso, PsoResult, StopReason};
    pub use crate::spaces::{ContinuousSpace, Discretization, IntegerSpace, MixedSpace, VarType};
    pub use crate::topology::{GlobalBest, Random, Ring, VonNeumann};
    pub use crate::traits::{BoundaryHandling, SearchSpace, Topology, UpdateContext, Velocity};
    pub use crate::velocity::{ConstrictionVelocity, FipsVelocity, InertiaVelocity};
}
