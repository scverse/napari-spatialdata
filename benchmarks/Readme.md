# Benchmarks

This are benchmarks for napari-spatialdata project.
Benchmarks are required for early detection of performance regressions.

Here we use `asv` to run benchmarks. Webpage https://asv.readthedocs.io

It is generally better to run benchmarks on linux runner (cheaper, more available).
However benchmarking of gui applications requires to have a display server.
For normal tests we use `xvfb` to run tests without display server,
but it impact performance of GUI applications in unpredictable way.
So we run GUI benchmarks on macOS runner.

To distinguish between gui and non-gui benchmarks we use `qt` in benchmark file name.

So any benchmark file starting with `benchamrk_qt` gui benchmark,
all other benchmarks are non-gui benchmarks.

## Running benchmarks

To run benchmarks locally there is a need to have `asv` installed and
python 3.11 available in the PATH.
First run requires to run `asv machine --yes` to collect machine metadata.

To quick run all benchmarks use `PR=1 asv run --show-stderr --quick  --attribute timeout=300 HEAD^!`
Quick run allow to check if benchmarks are running without errors.
`PR=1` environment variable is used to allow skip long running benchmarks cases.

To run all benchmarks and get more accurate statistics
use `asv run --show-stderr --attribute timeout=300 HEAD^!`

Each time you create a PR, benchmarks will be executed on the PR with PR=1 environment variable.

## Adding new benchmarks

To add a new benchmark first check if it fit into existing benchmark file.
If not create a new benchmark file, but remember to use `benchmark_qt` in file name prefix
if it is a gui benchmark.

## Maintaining benchmarks

To keep more reproducible results, the `asv` is pinning python to a specific
version in its configuration file (`asv.conf.json`).
If you want to update the python version it need to be bumped in both
`asv.conf.json` and `.github/workflows/benchmarks.yml` files.
