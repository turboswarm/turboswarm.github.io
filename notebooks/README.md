# Example notebooks

Runnable examples that tour the library. They need the compiled native module
(`maturin develop --release`) and `matplotlib`:

1. [`01_quickstart.ipynb`](01_quickstart.ipynb) — minimize a benchmark or your
   own function, read the result, plot convergence.
2. [`02_variants_and_topologies.ipynb`](02_variants_and_topologies.ipynb) —
   compare velocity variants and topologies.
3. [`03_integer_mixed_constraints.ipynb`](03_integer_mixed_constraints.ipynb) —
   integer, binary (knapsack), mixed variables and inequality constraints.
4. [`04_multiobjective.ipynb`](04_multiobjective.ipynb) — MOPSO and the Pareto
   front.
5. [`05_grey_numbers.ipynb`](05_grey_numbers.ipynb) — optimize grey numbers
   (interval / center+spread representation), bound each one, read the result.

```bash
pip install -e ".[notebooks]"   # jupyter, plotly
jupyter lab notebooks/
```
