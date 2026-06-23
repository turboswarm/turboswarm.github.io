//! Ring topology (lbest): each particle is informed only by its immediate
//! neighbors by index, in a circle. It propagates information more slowly
//! than `GlobalBest`, which reduces the risk of a local optimum at the cost
//! of slower convergence.

use crate::swarm::Swarm;
use crate::traits::Topology;

/// Ring with `each_side` neighbors on each side (a neighborhood of size
/// `2·each_side + 1`, counting itself). `each_side = 1` is the classic lbest
/// (left and right neighbors).
#[derive(Debug, Clone)]
pub struct Ring {
    each_side: usize,
}

impl Ring {
    /// Ring with `each_side` neighbors per side. Requires `each_side >= 1`.
    pub fn new(each_side: usize) -> Self {
        assert!(
            each_side >= 1,
            "the ring needs at least 1 neighbor per side"
        );
        Self { each_side }
    }

    /// The classic lbest ring (one neighbor on each side).
    pub fn lbest() -> Self {
        Self::new(1)
    }
}

impl Default for Ring {
    fn default() -> Self {
        Self::lbest()
    }
}

impl Topology for Ring {
    fn neighbors(&self, i: usize, swarm: &Swarm) -> Vec<usize> {
        let n = swarm.len();
        // Order [i, i+1, i-1, i+2, i-2, …]: the particle itself first, then
        // alternating right and left. This way the `best_among` tie-break
        // (first minimum) reproduces the classic scan. We avoid duplicate
        // indices (relevant for FIPS) in small swarms.
        let mut idx = Vec::with_capacity(2 * self.each_side + 1);
        idx.push(i);
        for off in 1..=self.each_side {
            let off = off % n; // rings with each_side >= n wrap around.
            if off == 0 {
                continue;
            }
            for j in [(i + off) % n, (i + n - off) % n] {
                if !idx.contains(&j) {
                    idx.push(j);
                }
            }
        }
        idx
    }
}
