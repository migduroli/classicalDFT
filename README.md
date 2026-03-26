# classicalDFT

A modern C++20 library for classical density functional theory (DFT) calculations.

## Requirements

- C++20 compiler (GCC 12+, Clang 15+, AppleClang 15+)
- CMake 3.20+
- [GSL](https://www.gnu.org/software/gsl/), [Boost](https://www.boost.org/) (serialization), [Armadillo](http://arma.sourceforge.net/)
- Docker (for running tests)
- Google Test is fetched automatically via CMake FetchContent

## Developer quickstart

```bash
make check       # verify dependencies
make build       # configure + compile
make test        # run tests in Docker
make test-local  # run tests locally (after build)
make format      # clang-format
make lint        # clang-tidy
make clean       # remove build artifacts
```

Run `make` with no arguments to see all available targets.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).