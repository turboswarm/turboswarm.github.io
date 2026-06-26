"""Expose turboswarm as a tool an LLM agent can call.

`optimization_tool()` returns a LangChain ``StructuredTool`` that an agent (in
LangChain or as a LangGraph ``ToolNode``) can invoke to minimize a standard
benchmark function with PSO. It is intentionally **safe**: the agent chooses a
*named* benchmark and a bounded budget, never arbitrary code.

    from langchain.agents import create_react_agent  # any LLM provider
    from turboswarm.integrations.agents import optimization_tool

    tools = [optimization_tool()]
    # ...wire `tools` into your agent / LangGraph graph...

Requires LangChain core (``pip install turboswarm[agents]``). The tool itself
calls no LLM and is provider-agnostic.
"""

from __future__ import annotations

import turboswarm as _ts

#: Benchmarks the tool is allowed to optimize (safe, no arbitrary code).
ALLOWED_BENCHMARKS = (
    "sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "schwefel",
    "bent_cigar", "discus", "elliptic", "zakharov", "levy", "expanded_schaffer",
)


def optimization_tool(name="minimize_benchmark", *, max_particles=200,
                      max_iterations=1000):
    """Build a LangChain ``StructuredTool`` that minimizes a benchmark with PSO.

    Args:
        name: the tool name exposed to the agent.
        max_particles, max_iterations: hard caps on the budget the agent may
            request (protects against runaway calls).

    Returns:
        A ``langchain_core.tools.StructuredTool``. Calling it returns a dict with
        ``best_position``, ``best_value``, ``evaluations`` and ``stop_reason``.
    """
    try:
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - exercised without langchain
        raise ImportError(
            "langchain-core is required for turboswarm.integrations.agents; "
            "install it with: pip install turboswarm[agents]"
        ) from exc

    allowed = ", ".join(ALLOWED_BENCHMARKS)

    class OptimizeArgs(BaseModel):
        benchmark: str = Field(
            description=f"name of the function to minimize, one of: {allowed}")
        dim: int = Field(ge=1, le=100,
                         description="number of dimensions of the problem")
        bound: float = Field(
            default=5.12, gt=0,
            description="symmetric bound; the search box is [-bound, bound] "
                        "per dimension")
        n_particles: int = Field(default=30, ge=2)
        max_iter: int = Field(default=100, ge=1)
        seed: int | None = Field(default=None,
                                 description="optional RNG seed for reproducibility")

    def _run(benchmark, dim, bound=5.12, n_particles=30, max_iter=100, seed=None):
        if benchmark not in ALLOWED_BENCHMARKS:
            raise ValueError(
                f"unknown benchmark '{benchmark}'. Allowed: {allowed}")
        result = _ts.minimize(
            benchmark, bounds=(-abs(bound), abs(bound)), dim=dim,
            n_particles=min(n_particles, max_particles),
            max_iter=min(max_iter, max_iterations),
            seed=seed, record_history=False,
        )
        return {
            "best_position": [float(v) for v in result.best_position],
            "best_value": float(result.best_value),
            "evaluations": result.evaluations,
            "stop_reason": result.stop_reason,
        }

    return StructuredTool.from_function(
        func=_run, name=name, args_schema=OptimizeArgs,
        description=(
            "Minimize a standard benchmark function with Particle Swarm "
            "Optimization (gradient-free). Returns the best position and value "
            "found. Use for numeric optimization subproblems."),
    )
