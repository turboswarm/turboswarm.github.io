#!/usr/bin/env bash
# Build the project documentation.
#
#   Portal (narrative + Python API)  ->  site/index.html   (MkDocs Material)
#   Rust API reference               ->  target/doc/turboswarm_core/index.html (rustdoc)
#
# Usage:
#   ./scripts/build-docs.sh            # build the portal and the Rust API
#   ./scripts/build-docs.sh --serve    # live-reloading portal at localhost:8000
#
# Requires the Python venv with the module installed (maturin develop) and the
# docs extra: pip install -e ".[docs]". The Rust API needs cargo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--serve" ]]; then
    echo ">> Serving the portal at http://127.0.0.1:8000 (Ctrl-C to stop)"
    exec mkdocs serve
fi

echo ">> Documentation portal (MkDocs)"
mkdocs build

echo ">> Rust API (rustdoc)"
cargo doc -p turboswarm-core --no-deps

echo ""
echo "Portal:   site/index.html"
echo "Rust API: target/doc/turboswarm_core/index.html"
