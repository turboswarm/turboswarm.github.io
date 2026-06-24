//! Grey test functions: benchmarks whose decision variables are grey numbers.
//!
//! Unlike the crisp benchmarks (`fn(&[f64]) -> f64`), these operate on
//! [`Grey`] numbers and return the whitenized scalar to minimize, so they can
//! validate [`GreySpace`](crate::spaces::GreySpace) and `minimize_grey`.

use crate::spaces::Grey;

/// Metadata for a grey test function.
#[derive(Debug, Clone)]
pub struct GreyBenchmark {
    /// Name of the function (used to dispatch it by string).
    pub name: &'static str,
    /// Recommended symmetric bound for the CENTER of each grey variable.
    pub center_bound: f64,
    /// Recommended maximum spread (half-width) of each grey variable.
    pub max_spread: f64,
    /// Value of the global optimum.
    pub optimum_value: f64,
}

/// Grey sphere: the expected (midpoint) sphere plus a unit penalty on the total
/// uncertainty. For grey numbers ⊗ᵢ with center `cᵢ` and spread `rᵢ`:
///
/// `f(⊗) = Σ cᵢ² + Σ rᵢ`.
///
/// Global minimum `f = 0` at the crisp origin (every ⊗ᵢ = `[0, 0]`): the
/// objective rewards both accuracy (centers at 0) and certainty (zero spread).
///
/// Note: the spread term is written explicitly rather than via interval
/// arithmetic. Squaring an interval as `⊗·⊗` would overestimate (the
/// dependency problem: `[−r, r]·[−r, r] = [−r², r²]`, whose center is 0
/// regardless of `r`), which would fail to penalize uncertainty and leave the
/// optimum non-unique.
pub fn grey_sphere(g: &[Grey]) -> f64 {
    g.iter()
        .map(|gi| gi.center() * gi.center() + gi.spread())
        .sum()
}

/// Metadata for [`grey_sphere`].
pub const GREY_SPHERE: GreyBenchmark = GreyBenchmark {
    name: "grey_sphere",
    center_bound: 5.12,
    max_spread: 5.12,
    optimum_value: 0.0,
};

/// All registered grey benchmarks with their metadata.
pub const GREY_ALL: &[GreyBenchmark] = &[GREY_SPHERE];

/// Looks up the metadata of a grey benchmark by name.
pub fn grey_meta(name: &str) -> Option<&'static GreyBenchmark> {
    GREY_ALL.iter().find(|b| b.name == name)
}
