//! Random topology: at every query, each particle is informed by `k` randomly
//! chosen neighbors (plus itself). The neighborhood changes from iteration to
//! iteration, which keeps information flow diverse — a useful middle ground
//! that does not depend on any fixed structure.
//!
//! The topology owns its own seeded RNG (independent of the optimizer's), so
//! runs are reproducible without disturbing the velocity RNG stream.

use std::cell::RefCell;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::swarm::Swarm;
use crate::traits::Topology;

/// Random topology with `k` informants per particle.
#[derive(Debug)]
pub struct Random {
    k: usize,
    rng: RefCell<ChaCha8Rng>,
}

impl Random {
    /// Creates the topology with `k` random informants per particle, using
    /// `seed` for its internal RNG (so the random neighborhoods are
    /// reproducible).
    pub fn new(k: usize, seed: u64) -> Self {
        assert!(k >= 1, "Random topology needs k >= 1 informants");
        Self {
            k,
            rng: RefCell::new(ChaCha8Rng::seed_from_u64(seed)),
        }
    }
}

impl Topology for Random {
    fn neighbors(&self, i: usize, swarm: &Swarm) -> Vec<usize> {
        let n = swarm.len();
        // The particle itself is always an informant; add up to `k` distinct
        // random others (capped by the swarm size).
        let want = self.k.min(n.saturating_sub(1));
        let mut idx = Vec::with_capacity(want + 1);
        idx.push(i);
        let mut rng = self.rng.borrow_mut();
        while idx.len() < want + 1 {
            let j = rng.gen_range(0..n);
            if !idx.contains(&j) {
                idx.push(j);
            }
        }
        idx
    }
}
