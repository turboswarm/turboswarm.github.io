//! Fully Informed Particle Swarm — FIPS (Mendes, Kennedy & Neves, 2004).
//!
//! Central idea: in classic PSO the particle listens to only ONE social
//! source (the best of its neighborhood). In FIPS it is *fully informed*:
//! ALL of its neighbors influence it, each with a weight. There is therefore no
//! separate cognitive term; the particle's own `pbest` enters as just another
//! neighbor (the `Topology::neighbors` convention includes it).
//!
//! v' = χ·[ v + Σ_{k∈N} U(0, φ/|N|)·(pbest_k − x) ]
//!
//! The total coefficient φ is distributed equally among the |N| neighbors, so
//! that the expected social acceleration (φ/2) is the same as in the
//! constriction variant: FIPS redistributes that "force", it does not increase
//! it. χ is the same Clerc-Kennedy constriction factor, derived from φ.
//!
//! It works better with *local* topologies (ring, Von Neumann): with
//! `GlobalBest` every particle is informed by every other and the swarm tends
//! to collapse early. That is the interesting contrast.

use rand::{Rng, RngCore};

use crate::traits::{UpdateContext, Velocity};

/// Fully informed velocity rule (FIPS).
#[derive(Debug, Clone)]
pub struct FipsVelocity {
    /// Total acceleration coefficient φ (distributed among the neighbors).
    pub phi: f64,
    /// Constriction factor, derived from `phi` at construction time.
    chi: f64,
}

impl FipsVelocity {
    /// Creates FIPS with total coefficient `phi`.
    ///
    /// # Panics
    /// If `phi <= 4`, where the constriction formula is not defined.
    pub fn new(phi: f64) -> Self {
        assert!(
            phi > 4.0,
            "FIPS uses constriction: requires φ > 4 (got {phi})"
        );
        let chi = 2.0 / (2.0 - phi - (phi * phi - 4.0 * phi).sqrt()).abs();
        Self { phi, chi }
    }

    /// The effective constriction factor χ (read-only).
    pub fn chi(&self) -> f64 {
        self.chi
    }
}

impl Default for FipsVelocity {
    /// Classic value φ = 4.1 (χ ≈ 0.7298), same as standard constriction.
    fn default() -> Self {
        Self::new(4.1)
    }
}

impl Velocity for FipsVelocity {
    fn update(&self, ctx: &UpdateContext, rng: &mut dyn RngCore) -> Vec<f64> {
        let dim = ctx.position.len();
        // |N|: number of informing neighbors. At least 1 to avoid dividing by 0.
        let k = ctx.neighbor_bests.len().max(1);
        let phi_k = self.phi / k as f64;

        let mut new_v = Vec::with_capacity(dim);
        for d in 0..dim {
            // Aggregate attraction toward each neighbor's pbest.
            let mut social = 0.0;
            for nb in ctx.neighbor_bests {
                let r: f64 = rng.gen();
                social += phi_k * r * (nb[d] - ctx.position[d]);
            }
            new_v.push(self.chi * (ctx.velocity[d] + social));
        }
        new_v
    }

    /// FIPS is fully informed: it reads every neighbor's `pbest`.
    fn needs_full_neighborhood(&self) -> bool {
        true
    }
}
