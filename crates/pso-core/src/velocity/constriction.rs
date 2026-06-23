//! PSO with constriction factor (Clerc & Kennedy, 2002).
//!
//! v' = χ·[ v + c1·r1·(pbest − x) + c2·r2·(nbest − x) ]
//!
//! where the constriction factor χ is derived from the coefficients:
//!
//! χ = 2 / |2 − φ − √(φ² − 4φ)|,   with φ = c1 + c2  and  φ > 4.
//!
//! With the classic values c1 = c2 = 2.05 (φ = 4.1) you get χ ≈ 0.7298, which
//! is exactly the `w = 0.729` that the inertia variant uses by default: both
//! formulations are mathematically equivalent at that point. The
//! difference is that here χ is *derived* from the convergence guarantee
//! instead of being set by hand.

use rand::{Rng, RngCore};

use crate::traits::{UpdateContext, Velocity};

/// Velocity rule with constriction factor.
#[derive(Debug, Clone)]
pub struct ConstrictionVelocity {
    /// Cognitive coefficient (attraction to the personal best).
    pub c1: f64,
    /// Social coefficient (attraction to the neighborhood best).
    pub c2: f64,
    /// Constriction factor, derived from `c1 + c2` at construction time.
    chi: f64,
}

impl ConstrictionVelocity {
    /// Creates the rule from the cognitive/social coefficients.
    ///
    /// # Panics
    /// If `c1 + c2 <= 4`, where the constriction formula is not defined.
    pub fn new(c1: f64, c2: f64) -> Self {
        let phi = c1 + c2;
        assert!(
            phi > 4.0,
            "constriction requires c1 + c2 > 4 (got {phi})"
        );
        let chi = 2.0 / (2.0 - phi - (phi * phi - 4.0 * phi).sqrt()).abs();
        Self { c1, c2, chi }
    }

    /// The effective constriction factor χ (read-only).
    pub fn chi(&self) -> f64 {
        self.chi
    }
}

impl Default for ConstrictionVelocity {
    /// Classic Clerc-Kennedy values: c1 = c2 = 2.05 (χ ≈ 0.7298).
    fn default() -> Self {
        Self::new(2.05, 2.05)
    }
}

impl Velocity for ConstrictionVelocity {
    fn update(&self, ctx: &UpdateContext, rng: &mut dyn RngCore) -> Vec<f64> {
        let dim = ctx.position.len();
        let mut new_v = Vec::with_capacity(dim);

        for d in 0..dim {
            let r1: f64 = rng.gen();
            let r2: f64 = rng.gen();
            let cognitive = self.c1 * r1 * (ctx.personal_best[d] - ctx.position[d]);
            let social = self.c2 * r2 * (ctx.neighbor_best[d] - ctx.position[d]);
            new_v.push(self.chi * (ctx.velocity[d] + cognitive + social));
        }
        new_v
    }
}
