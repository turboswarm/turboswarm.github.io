//! Social topologies: they define the neighborhood of each particle.
//!
//! Phase 1: `GlobalBest` implemented.
//! Phase 2: `Ring` (lbest ring) and `VonNeumann` (2D grid) implemented.

mod global;
mod random;
mod ring;
mod von_neumann;

pub use global::GlobalBest;
pub use random::Random;
pub use ring::Ring;
pub use von_neumann::VonNeumann;
