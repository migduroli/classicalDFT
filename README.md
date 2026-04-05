# Modern classicalDFT

A modern C++23 library for classical density functional theory.

## Requirements

- C++23 compiler (GCC 13+, Clang 17+)
- CMake 3.20+
- [Armadillo](http://arma.sourceforge.net/), [FFTW3](https://www.fftw.org/), [GSL](https://www.gnu.org/software/gsl/), [nlohmann/json](https://github.com/nlohmann/json)
- [autodiff](https://github.com/autodiff/autodiff) and [Catch2](https://github.com/catchorg/Catch2) are fetched automatically via CMake FetchContent

## Developer quickstart

```bash
make check              # verify dependencies
make build              # configure + compile (Docker)
make test               # run all tests in Docker (with coverage)
make unit-tests         # run unit tests in Docker (with coverage)
make integration-tests  # run integration tests in Docker (with coverage)
make format             # clang-format (include, src, tests, docs)
make lint               # clang-tidy
make clean              # remove build artifacts
```

### Local build (no Docker)

```bash
cmake -S . -B build-local -DCMAKE_CXX_COMPILER=g++-15
cmake --build build-local --parallel
ctest --test-dir build-local --output-on-failure
```

Run `make` with no arguments to see all available targets.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).