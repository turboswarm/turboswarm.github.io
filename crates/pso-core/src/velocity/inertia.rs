//! PSO with inertia weight (Shi & Eberhart, 1998).
//!
//! v' = w·v + c1·r1·(pbest - x) + c2·r2·(nbest - x)

use rand::{Rng, RngCore};

use crate::traits::{UpdateContext, Velocity};

/// Velocity rule with inertia weight.
///
/// Optionally, the weight `w` can decay linearly from `w` to `w_end`
/// over the iterations (a classic exploration/exploitation balancing
/// strategy).
#[derive(Debug, Clone)]
pub struct InertiaVelocity {
    /// Inertia weight (how much of the previous velocity is preserved).
    pub w: f64,
    /// Cognitive coefficient (attraction to the personal best).
    pub c1: f64,
    /// Social coefficient (attraction to the neighborhood best).
    pub c2: f64,
    /// If `Some`, the weight decays linearly down to this value.
    pub w_end: Option<f64>,
}

impl InertiaVelocity {
    /// Creates the rule with inertia weight `w` and coefficients `c1`, `c2`.
    pub fn new(w: f64, c1: f64, c2: f64) -> Self {
        Self { w, c1, c2, w_end: None }
    }

    /// Enables linear decay of the inertia weight (chainable).
    pub fn with_decay(mut self, w_end: f64) -> Self {
        self.w_end = Some(w_end);
        self
    }

    fn current_w(&self, iter: usize, max_iter: usize) -> f64 {
        match self.w_end {
            Some(w_end) if max_iter > 1 => {
                let t = iter as f64 / (max_iter - 1) as f64;
                self.w + (w_end - self.w) * t
            }
            _ => self.w,
        }
    }
}

impl Velocity for InertiaVelocity {
    fn update(&self, ctx: &UpdateContext, rng: &mut dyn RngCore) -> Vec<f64> {
        let w = self.current_w(ctx.iteration, ctx.max_iterations);
        let dim = ctx.position.len();
        let mut new_v = Vec::with_capacity(dim);

        for d in 0..dim {
            let r1: f64 = rng.gen();
            let r2: f64 = rng.gen();
            let cognitive = self.c1 * r1 * (ctx.personal_best[d] - ctx.position[d]);
            let social = self.c2 * r2 * (ctx.neighbor_best[d] - ctx.position[d]);
            new_v.push(w * ctx.velocity[d] + cognitive + social);
        }
        new_v
    }
}
