# Boundary handling

When a particle steps outside the search box, the optimizer brings it back. How
it does that is controlled by `bounds_handling`; each strategy changes the
search dynamics near the edges.

| Strategy | What it does |
|----------|--------------|
| `"clamp"` (default) | Clip the position to the boundary. Simple; particles can pile up on the edge. |
| `"reflect"` | Bounce the position back into range and flip the offending velocity component. |
| `"wrap"` | Wrap around toroidally to the opposite side — useful for periodic domains. |
| `"reinit"` | Re-sample the offending component uniformly within bounds and zero its velocity (adds diversity). |

```python
import turboswarm as pso

r = pso.minimize("rastrigin", bounds=[(-5.12, 5.12)] * 2,
                 bounds_handling="reflect", seed=1)
```

All strategies guarantee the reported `best_position` lies within the bounds.
The default is `"clamp"`, so existing behavior is unchanged unless you opt in.

## From Rust

The strategy lives in `PsoParams`; the built-in spaces implement all four. A
custom `SearchSpace` that does not override `enforce_bounds` falls back to
clamping.

```rust
use pso_core::prelude::*;

let params = PsoParams {
    bounds_handling: BoundaryHandling::Reflect,
    ..Default::default()
};
```
