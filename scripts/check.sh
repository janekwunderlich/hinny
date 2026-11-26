#!/bin/sh
# Every check CI would run, on this machine: the Rust tests, the fault sweep, the Python bindings
# (rebuilt into bindings/python/.venv), the Node binding and the benchmark protocol. Stops at the first failure.
# A GitHub Actions workflow can run exactly this once there are minutes for it.
set -eu
cd "$(dirname "$0")/.."

echo "== Rust"
cargo test --release --workspace --exclude hinny-python

echo "== C interface (a C program linked against the static library)"
sh crates/hinny-c/tests/run-c.sh

echo "== Faults (every I/O failure, then a crash)"
cargo test --release -p hinny --features failpoints --test faults --test fork

echo "== Python"
VENV=bindings/python/.venv
if [ ! -x "$VENV/bin/maturin" ]; then
    echo "No $VENV with maturin. Make one: python3 -m venv $VENV && $VENV/bin/pip install maturin numpy" >&2
    exit 1
fi
VIRTUAL_ENV="$PWD/$VENV" "$VENV/bin/maturin" develop --profile python --manifest-path bindings/python/Cargo.toml --quiet
"$VENV/bin/python" -I -m unittest discover -s bindings/python/tests

echo "== Node"
if command -v npm >/dev/null 2>&1; then
    (cd bindings/node && npm ci --silent && npm run build --silent && npm test --silent)
else
    echo "No npm here: the Node binding's tests are skipped."
fi

echo "== Benchmark protocol"
"$VENV/bin/python" -m unittest discover -s benchmarks/beir -p 'test_*.py'

echo "All checks passed."
