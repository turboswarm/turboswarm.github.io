//! Grey search space (prototype).
//!
//! A *grey number* ⊗ is a quantity known only to lie within an interval
//! `[a, b]`. Here each grey variable is encoded internally as a **center +
//! spread** pair `(c, r)` with `r ≥ 0`, so `[a, b] = [c − r, c + r]`. This
//! encoding is deliberately chosen over the raw interval `[a, b]`: the swarm
//! moves each internal coordinate independently, and `(c, r)` keeps the two
//! coordinates decoupled (center anywhere in range, spread non-negative),
//! whereas `[a, b]` would impose the coupled constraint `a ≤ b` that the swarm
//! would violate every step and force repairs.
//!
//! Approach (mirrors [`IntegerSpace`](super::IntegerSpace)): the swarm lives in
//! an internal CONTINUOUS space of `2 · n` coordinates (center and spread of
//! each of the `n` grey variables, interleaved as `[c₀, r₀, c₁, r₁, …]`), and
//! the grey reconstruction happens only at [`decode`](GreySpace::decode), at
//! evaluation time. The objective receives `&[Grey]` and is responsible for the
//! grey arithmetic and the whitenization (collapsing a grey result to the `f64`
//! the swarm compares).

use rand::{Rng, RngCore};

use crate::traits::SearchSpace;

/// A grey number ⊗ = `[lower, upper]`, stored as center + non-negative spread.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grey {
    center: f64,
    spread: f64,
}

impl Grey {
    /// Builds a grey number from a center and a spread (the spread is taken as
    /// `|spread|`, so it is always non-negative).
    pub fn new(center: f64, spread: f64) -> Self {
        Self {
            center,
            spread: spread.abs(),
        }
    }

    /// Builds a grey number from its interval `[lower, upper]`.
    /// Panics if `lower > upper`.
    pub fn from_interval(lower: f64, upper: f64) -> Self {
        assert!(lower <= upper, "invalid grey interval: {lower} > {upper}");
        Self {
            center: 0.5 * (lower + upper),
            spread: 0.5 * (upper - lower),
        }
    }

    /// The center (mean) of the interval.
    pub fn center(&self) -> f64 {
        self.center
    }

    /// The (non-negative) half-width of the interval.
    pub fn spread(&self) -> f64 {
        self.spread
    }

    /// Lower bound `a = c − r`.
    pub fn lower(&self) -> f64 {
        self.center - self.spread
    }

    /// Upper bound `b = c + r`.
    pub fn upper(&self) -> f64 {
        self.center + self.spread
    }

    /// Whitenization: a crisp representative `a + λ·(b − a)` for `λ ∈ [0, 1]`.
    /// `λ = 0.5` returns the center; the objective uses this to collapse a grey
    /// quantity to a comparable `f64`.
    pub fn whiten(&self, lambda: f64) -> f64 {
        self.lower() + lambda * (self.upper() - self.lower())
    }
}

// --- Grey (interval) arithmetic ---
// Standard interval arithmetic, so objectives can be written naturally over
// grey quantities. The result of every operation is a valid grey number
// (`lower ≤ upper`); `Grey::new` re-normalizes the spread to be non-negative.

impl std::ops::Add for Grey {
    type Output = Grey;
    /// `[a₁,b₁] + [a₂,b₂] = [a₁+a₂, b₁+b₂]`.
    fn add(self, rhs: Grey) -> Grey {
        Grey::from_interval(self.lower() + rhs.lower(), self.upper() + rhs.upper())
    }
}

impl std::ops::Sub for Grey {
    type Output = Grey;
    /// `[a₁,b₁] − [a₂,b₂] = [a₁−b₂, b₁−a₂]`.
    fn sub(self, rhs: Grey) -> Grey {
        Grey::from_interval(self.lower() - rhs.upper(), self.upper() - rhs.lower())
    }
}

impl std::ops::Mul for Grey {
    type Output = Grey;
    /// `[a₁,b₁] · [a₂,b₂] = [min P, max P]` over the four endpoint products `P`.
    fn mul(self, rhs: Grey) -> Grey {
        let (a1, b1, a2, b2) = (self.lower(), self.upper(), rhs.lower(), rhs.upper());
        let ps = [a1 * a2, a1 * b2, b1 * a2, b1 * b2];
        let lo = ps.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = ps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Grey::from_interval(lo, hi)
    }
}

impl std::ops::Mul<f64> for Grey {
    type Output = Grey;
    /// Scales an interval by a crisp factor (handles a negative factor, which
    /// flips the endpoints).
    fn mul(self, k: f64) -> Grey {
        let (p, q) = (self.lower() * k, self.upper() * k);
        Grey::from_interval(p.min(q), p.max(q))
    }
}

/// Grey search space: `n` grey variables, each constrained to lie within an
/// interval `[lower, upper]`. The swarm searches over the center and spread of
/// each grey number, and the whole decoded interval `[c − r, c + r]` is kept
/// inside `[lower, upper]`; an optional `max_spread` caps the half-width
/// further.
#[derive(Debug, Clone)]
pub struct GreySpace {
    /// `(lower, upper)` limits each grey variable's interval must stay within.
    bounds: Vec<(f64, f64)>,
    /// Extra cap on the spread (half-width) of each grey variable.
    /// `f64::INFINITY` means "limited only by the `bounds` box".
    max_spread: Vec<f64>,
}

impl GreySpace {
    /// Creates the space from per-variable interval limits `(lower, upper)` and
    /// an extra spread cap. Each grey number's whole interval is kept within its
    /// `(lower, upper)` limits, and its half-width is additionally capped by
    /// `max_spread` (use `f64::INFINITY` for "no extra cap", i.e. limited only
    /// by the `(lower, upper)` box).
    ///
    /// Panics if the two vectors differ in length, if any limit has
    /// `lower > upper`, or if any maximum spread is negative.
    pub fn new(bounds: Vec<(f64, f64)>, max_spread: Vec<f64>) -> Self {
        assert_eq!(
            bounds.len(),
            max_spread.len(),
            "bounds and max_spread must have the same length"
        );
        for (i, ((lo, hi), s)) in bounds.iter().zip(&max_spread).enumerate() {
            assert!(lo <= hi, "invalid bound in variable {i}: {lo} > {hi}");
            assert!(*s >= 0.0, "negative max spread in variable {i}: {s}");
        }
        Self { bounds, max_spread }
    }

    /// Convenience: `n` grey variables sharing the same `(lo, hi)` interval
    /// limits and the same `max_spread` cap.
    pub fn uniform(n: usize, lo: f64, hi: f64, max_spread: f64) -> Self {
        Self::new(vec![(lo, hi); n], vec![max_spread; n])
    }

    /// The `(lower, upper)` interval limits per grey variable.
    pub fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    /// Number of grey variables (half the internal continuous dimension).
    pub fn n_grey(&self) -> usize {
        self.bounds.len()
    }

    /// Largest feasible spread for grey variable `var` at center `c`: the
    /// distance to the nearest limit, capped by `max_spread`. Keeping the
    /// spread within `[0, spread_cap]` guarantees the decoded interval
    /// `[c − r, c + r]` stays within `(lower, upper)`.
    fn spread_cap(&self, var: usize, c: f64) -> f64 {
        let (lo, hi) = self.bounds[var];
        (c - lo).min(hi - c).min(self.max_spread[var]).max(0.0)
    }

    /// Largest spread the variable can ever take (at its mid-limit), used to
    /// scale the initial spread velocity and report the spread's span.
    fn max_feasible_spread(&self, var: usize) -> f64 {
        let (lo, hi) = self.bounds[var];
        (0.5 * (hi - lo)).min(self.max_spread[var]).max(0.0)
    }
}

impl SearchSpace for GreySpace {
    type Scalar = Grey;

    fn dim(&self) -> usize {
        // Two internal coordinates (center, spread) per grey variable.
        2 * self.bounds.len()
    }

    fn sample(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        let mut raw = Vec::with_capacity(self.dim());
        for var in 0..self.bounds.len() {
            let (lo, hi) = self.bounds[var];
            let c = rng.gen_range(lo..=hi);
            let cap = self.spread_cap(var, c);
            raw.push(c);
            raw.push(rng.gen_range(0.0..=cap.max(0.0)));
        }
        raw
    }

    fn sample_velocity(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        let mut vel = Vec::with_capacity(self.dim());
        for var in 0..self.bounds.len() {
            let (lo, hi) = self.bounds[var];
            let crange = hi - lo;
            vel.push(rng.gen_range(-crange..=crange));
            let srange = self.max_feasible_spread(var);
            vel.push(rng.gen_range(-srange..=srange.max(0.0)));
        }
        vel
    }

    /// Projects each grey variable onto its feasible region: the center is
    /// clamped to `(lower, upper)` and the spread to `[0, spread_cap(center)]`,
    /// so the decoded interval is always contained in `(lower, upper)`. Because
    /// the constraint couples center and spread, grey bounds are enforced by
    /// this projection regardless of the chosen `BoundaryHandling`.
    fn clamp(&self, position: &mut [f64]) {
        for var in 0..self.bounds.len() {
            let (lo, hi) = self.bounds[var];
            let c = position[2 * var].clamp(lo, hi);
            position[2 * var] = c;
            let cap = self.spread_cap(var, c);
            position[2 * var + 1] = position[2 * var + 1].clamp(0.0, cap);
        }
    }

    // No `enforce_bounds` override: the trait's default calls `clamp`, which is
    // the coupled projection above. The boundary-handling strategies (reflect /
    // wrap / reinit) don't apply to the triangular feasible region.

    fn decode(&self, raw: &[f64]) -> Vec<Grey> {
        raw.chunks_exact(2)
            .map(|cs| Grey::new(cs[0], cs[1]))
            .collect()
    }

    fn span(&self) -> Vec<(f64, f64)> {
        // Per internal coordinate: center spans (lower, upper); spread spans
        // [0, max feasible spread].
        let mut s = Vec::with_capacity(self.dim());
        for var in 0..self.bounds.len() {
            s.push(self.bounds[var]);
            s.push((0.0, self.max_feasible_spread(var)));
        }
        s
    }
}
