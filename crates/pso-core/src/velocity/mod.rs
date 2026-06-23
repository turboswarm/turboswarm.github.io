//! Velocity update rules. Each one is a PSO variant.
//!
//! Phase 1: `Inertia` implemented and tested.
//! Phase 2: `Constriction` (Clerc-Kennedy) and `Fips` (fully informed).
//!          `Fips` reads `UpdateContext::neighbor_bests` (the whole neighborhood).

mod constriction;
mod fips;
mod inertia;

pub use constriction::ConstrictionVelocity;
pub use fips::FipsVelocity;
pub use inertia::InertiaVelocity;
