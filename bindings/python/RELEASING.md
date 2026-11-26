# Releasing the Python package

Nothing here publishes on its own. The package is `hinny` on PyPI (the name is free as of
October 2026). Its version comes from `bindings/python/Cargo.toml`.

## What gets built

- **Wheels** for macOS on Apple silicon (`macosx_11_0_arm64`) and Intel (`macosx_10_12_x86_64`,
  cross-compiled), and Linux on x86_64 and aarch64 (`manylinux_2_28`, built in maturin's
  manylinux containers). Each uses Python's stable ABI (`cp311-abi3`), so one wheel per platform
  installs on CPython 3.11 and every later version. The macOS arm64 wheel is about 460 KB and
  links only system libraries.
- **A source distribution** (`hinny-<version>.tar.gz`), for other platforms: `pip` builds it with
  Rust 1.89 or later.

Both use Cargo's `python` profile (in the workspace `Cargo.toml`): the release profile, except
that a Rust panic unwinds and reaches Python as an exception instead of ending the process.
`maturin build` and `pip install` pick it up from `pyproject.toml`; `maturin build --release`
would use the release profile instead, so leave `--release` off.

## Building locally

```bash
cd bindings/python
maturin build --out dist
```

```bash
maturin sdist --out dist
```

To check the wheel in a clean environment (from outside the source tree, so the installed copy is
the one imported):

```bash
uv venv /tmp/hinny-check -p 3.11 && VIRTUAL_ENV=/tmp/hinny-check uv pip install dist/*.whl numpy
```

```bash
cd /tmp && /tmp/hinny-check/bin/python -m unittest discover -s "$OLDPWD/tests"
```

## In CI

`.github/workflows/python-wheels.yml` builds the four wheels and the sdist, tests the macOS arm64
and Linux x86_64 wheels on Python 3.11 and the newest release, builds and tests the sdist on
Linux, and keeps everything as one `dist` artifact. It runs
when a `py-v*` tag is pushed, or by hand: `gh workflow run python-wheels.yml`.

## Publishing (when it's time)

1. Bump `version` in `bindings/python/Cargo.toml`, commit, tag `py-v<version>`, push the tag.
2. Download the `dist` artifact from the workflow run.
3. Upload to TestPyPI first, install from there in a clean environment, then upload to PyPI:
   `twine upload dist/*` (or `uv publish`). Better: set up PyPI
   [trusted publishing](https://docs.pypi.org/trusted-publishers/) for this repository and add a
   `publish` job to the workflow, so no token is ever stored.

## Not yet

- **Tests on Intel Macs and Linux aarch64:** those wheels are built but CI has no runners to run
  them on; macOS arm64 and Linux x86_64 wheels are tested on Python 3.11 and the newest.
- **Windows:** untested. (The file-name index is macOS-only, but it's an optional feature the
  bindings don't use.)
