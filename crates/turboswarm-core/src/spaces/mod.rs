//! Search spaces: they define the domain and the integer/real distinction.

mod continuous;
mod integer;
mod mixed;

pub use continuous::ContinuousSpace;
pub use integer::{Discretization, IntegerSpace};
pub use mixed::{MixedSpace, VarType};

use rand::{Rng, RngCore};

use crate::traits::BoundaryHandling;

/// Applies a boundary-handling strategy per dimension to a particle whose
/// position is in `[lo, hi]` per component (given by `bound(i)`). Shared by the
/// built-in spaces. Only out-of-range components are touched.
pub(crate) fn apply_boundary(
    position: &mut [f64],
    velocity: &mut [f64],
    bound: impl Fn(usize) -> (f64, f64),
    handling: BoundaryHandling,
    rng: &mut dyn RngCore,
) {
    for i in 0..position.len() {
        let (lo, hi) = bound(i);
        let x = position[i];
        if x >= lo && x <= hi {
            continue;
        }
        let range = hi - lo;
        match handling {
            BoundaryHandling::Clamp => position[i] = x.clamp(lo, hi),
            BoundaryHandling::Reflect => {
                if range <= 0.0 {
                    position[i] = lo;
                } else {
                    let mut nx = x;
                    while nx < lo || nx > hi {
                        if nx < lo {
                            nx = lo + (lo - nx);
                        }
                        if nx > hi {
                            nx = hi - (nx - hi);
                        }
                    }
                    position[i] = nx;
                    velocity[i] = -velocity[i];
                }
            }
            BoundaryHandling::Wrap => {
                if range <= 0.0 {
                    position[i] = lo;
                } else {
                    let mut nx = (x - lo) % range;
                    if nx < 0.0 {
                        nx += range;
                    }
                    position[i] = lo + nx;
                }
            }
            BoundaryHandling::Reinit => {
                position[i] = if range <= 0.0 {
                    lo
                } else {
                    rng.gen_range(lo..=hi)
                };
                velocity[i] = 0.0;
            }
        }
    }
}
