//! Global topology (gbest): every particle is informed by the global
//! best of the swarm. Fast convergence, higher risk of a local optimum.

use crate::swarm::Swarm;
use crate::traits::Topology;

/// Global topology (gbest): every particle sees the best of the swarm.
#[derive(Debug, Clone, Default)]
pub struct GlobalBest;

impl GlobalBest {
    /// Creates the global topology.
    pub fn new() -> Self {
        Self
    }
}

impl Topology for GlobalBest {
    /// The neighborhood is the WHOLE swarm: each particle sees all the others.
    fn neighbors(&self, _i: usize, swarm: &Swarm) -> Vec<usize> {
        (0..swarm.len()).collect()
    }
}
