"""Optional integrations with popular libraries.

Each submodule imports its third-party dependency lazily (inside the functions
that need it), so ``import turboswarm`` and ``import turboswarm.integrations``
never require scipy/pandas to be installed. Install the relevant extra when you
use one, e.g.::

    pip install turboswarm[scipy]      # SciPy drop-in wrapper
    pip install turboswarm[pandas]     # history/convergence DataFrames
    pip install turboswarm[all]        # everything

Submodules:
  - ``scipy``  : :func:`minimize` mirroring ``scipy.optimize.minimize``.
  - ``pandas`` : export an optimization history/convergence as a ``DataFrame``.
"""

from . import pandas, scipy  # noqa: F401  (lazy third-party imports inside)

__all__ = ["scipy", "pandas"]
