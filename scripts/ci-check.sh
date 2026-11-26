#!/usr/bin/env bash
# Checks run by CI on Linux, macOS and Windows (and locally, with no hardware).
# Requires: uv, cargo with the wasm32-unknown-unknown target, and the Dyne submodule
# (git submodule update --init).
set -euo pipefail
cd "$(dirname "$0")/.."

test -f third_party/dyne/scripts/onnx_import.py || {
  echo "Dyne submodule missing: run 'git submodule update --init'" >&2; exit 1; }

uv sync --locked
cargo test --quiet --locked --manifest-path crates/erg-links/Cargo.toml --target-dir target/erg-links-test
uv run pytest -q
echo "ci-check: PASS"
