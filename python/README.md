# turboswarm

**Particle Swarm Optimization** with a compute core in **Rust** and an API in
**Python**. Focused on visualization, variant comparison and clear code.
Supports real, integer, binary and mixed variables, constraints and
multi-objective optimization.

## Installation

From source (development), with [maturin](https://www.maturin.rs/):

```bash
pip install maturin
python -m venv .venv && source .venv/bin/activate
maturin develop --release      # compiles the Rust core and installs it
```

## Usage

```python
import turboswarm as pso

# Native benchmark (fast, in Rust, without the GIL)
r = pso.minimize("rastrigin", bounds=(-5.12, 5.12), dim=2, seed=42)

# Your own function in Python
r = pso.minimize(lambda x: sum(xi**2 for xi in x), bounds=(-5, 5), dim=3)

# Integer variables
r = pso.minimize(f, bounds=(-10, 10), dim=2, integer=True)

# Variant and topology by name
r = pso.minimize("ackley", bounds=(-32.768, 32.768), dim=2,
                 velocity="fips", topology="ring", seed=1)

print(r.best_position, r.best_value)
```

### Parameters of `minimize`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `objective` | — | callable `f(list)->float`, or name of a native benchmark |
| `bounds` | — | list of `(min, max)` per dimension |
| `integer` / `binary` | `False` | optimize over integers / `{0,1}` |
| `var_types` | `None` | per-dimension `"real"`/`"integer"`/`"binary"` (mixed) |
| `n_particles` | `30` | swarm size |
| `max_iter` | `100` | iterations |
| `w, c1, c2` | `0.729, 1.494, 1.494` | inertia, cognitive, social |
| `velocity` | `"inertia"` | `"inertia"`, `"constriction"`, `"fips"` |
| `topology` | `"global"` | `"global"`, `"ring"`, `"vonneumann"`, `"random"` |
| `bounds_handling` | `"clamp"` | `"clamp"`, `"reflect"`, `"wrap"`, `"reinit"` |
| `seed` | `None` | seed (fix it for reproducibility) |
| `record_history` | `True` | store the trace for visualization |
| `v_max` | `None` | clamp each velocity component to `[-v_max, v_max]` |
| `patience` / `tol` | `0` / `0.0` | stop after `patience` iters without `>tol` improvement |
| `max_evals` / `target` / `max_time` | `None` | stop on evaluation / value / time budget |
| `constraints` / `penalty` | `None` / `1e6` | inequality constraints `g(x)<=0` via penalty |
| `callback` | `None` | `callback(iteration, best_value)`; return `False` to stop |
| `vectorized` | `False` | objective receives the whole swarm as a NumPy array |

**Native benchmarks:** `sphere`, `rastrigin`, `rosenbrock`, `ackley`,
`griewank`, `schwefel`. Their metadata (recommended bound and optimum) are in
`pso.benchmark_info(name) -> (bound, optimum)`.

> FIPS performs better with local topologies (`"ring"`, `"vonneumann"`). The
> `"constriction"` and `"fips"` variants derive their factor from `c1 + c2`.

### Result (`PsoResult`)

- `best_position` — list of floats (whole-valued for integer/binary dims)
- `best_value` — float
- `convergence` — best value per iteration (convergence curve)
- `history` — `history[iter][particle][dim]` (empty if `record_history=False`)
- `evaluations` — number of objective evaluations performed
- `stop_reason` — `"max_iterations"`, `"target"`, `"max_evaluations"`,
  `"stagnation"`, `"max_time"` or `"callback"`

### Multi-objective (MOPSO)

`minimize_multi` returns a `ParetoFront` (`.positions`, `.objectives`):

```python
front = pso.minimize_multi(
    lambda x: [sum(xi**2 for xi in x), sum((xi - 2) ** 2 for xi in x)],
    bounds=[(-5, 5)] * 2, seed=42,
)
print(len(front))            # non-dominated solutions
```

## Visualization

```python
import matplotlib.pyplot as plt

pso.viz.plot_convergence(r); plt.show()
pso.viz.compare({"inertia": rA, "fips": rB}); plt.show()
pso.viz.plot_pareto(front); plt.show()   # objective space of a Pareto front

anim = pso.viz.animate_swarm(r, pso.benchmarks.rastrigin, [(-5.12, 5.12)] * 2)
# in a notebook:  from IPython.display import HTML; HTML(anim.to_jshtml())
```

`animate_swarm` only supports 2D problems and requires `record_history=True`.

## Documentation

A navigable documentation portal (narrative guide + API reference) is built with
MkDocs Material:

```bash
pip install -e ".[docs]"
./scripts/build-docs.sh --serve   # http://127.0.0.1:8000
```

## License

MIT.
