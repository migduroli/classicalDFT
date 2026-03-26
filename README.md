# classicalDFT

A modern C++20 library for classical density functional theory (DFT) calculations.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build -V
```

### Requirements

- C++20 compiler (GCC 12+, Clang 15+, AppleClang 15+)
- CMake 3.20+
- [GSL](https://www.gnu.org/software/gsl/) (GNU Scientific Library)
- [Boost](https://www.boost.org/) (serialization, property_tree)
- [Armadillo](http://arma.sourceforge.net/) (linear algebra)
- [Grace](http://plasma-gate.weizmann.ac.il/Grace/) (optional, for plotting)
- Google Test is fetched automatically via CMake FetchContent

### CMake options

| Option | Default | Description |
|---|---|---|
| `DFT_BUILD_TESTS` | `ON` | Build unit tests |
| `DFT_BUILD_EXAMPLES` | `ON` | Build example programs |
| `DFT_USE_GRACE` | `ON` | Enable xmgrace plotting |

## Structure

```
include/dft_lib/       Headers (public API)
src/                   Implementation files
tests/                 Google Test unit tests
examples/              Example programs
cmake/                 CMake modules
legacy_lib/            Legacy reference code
documentation/         Installation guides, LaTeX notes
```

## Modules

| Module | Namespace | Description |
|---|---|---|
| `utils/console` | `console` | Terminal I/O with colour support |
| `utils/config_parser` | `dft_core::config_parser` | INI/JSON/XML/INFO config file reader |
| `utils/functions` | `dft_core::utils::functions` | Vectorised function application |
| `exceptions` | `dft_core::exception` | Grace and parameter exceptions |
| `graph/grace` | `dft_core::grace_plot` | xmgrace plotting wrapper |
| `numerics/arithmetic` | `dft_core::numerics::arithmetic` | Compensated summation (Kahan-Babuska) |
| `numerics/integration` | `dft_core::numerics::integration` | GSL numerical integration |
| `physics/potentials` | `dft_core::physics::potentials` | Intermolecular potentials (LJ, tWF, WRDF) |
| `geometry` | `dft_core::geometry` | Vertex, Element, Mesh (2D/3D) |

## License

See [LICENSE](LICENSE).

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details

## Acknowledgments

Development was funded by ESA