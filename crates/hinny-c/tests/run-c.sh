#!/bin/sh
# Builds the static library and tests/smoke.c against it with the system's C compiler, and runs
# it: the interface as a C program meets it (the header compiles, the symbols link, the calls
# work). From the repository root: sh crates/hinny-c/tests/run-c.sh
set -eu
cd "$(dirname "$0")/../../.."
cargo build --release -p hinny-c
target=${CARGO_TARGET_DIR:-target}
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
case "$(uname)" in
    Darwin) libs="-framework Accelerate -lpthread" ;;
    *) libs="-lpthread -ldl -lm" ;;
esac
${CC:-cc} -Wall -Wextra -Werror -std=c11 -I crates/hinny-c/include crates/hinny-c/tests/smoke.c "$target/release/libhinny_c.a" $libs -o "$out/smoke"
"$out/smoke" "$out/data"
