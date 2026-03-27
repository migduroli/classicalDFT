# classicalDFT reinvention: action plan

## Status quo

### What happened

My last merged PR was **#30** (`9747248`, 25 April 2021), which completed the refactoring of `Lattice.h` into `geometry/mesh` with proper 2D/3D abstractions, full test coverage, and examples. At that point, the repository had a clean separation:

- `dft_lib/` — my modern C++14 library (`dft_core` namespace, GTest coverage, examples, Doxygen-ready doc-strings)
- `legacy_lib/` — Jim's original code, untouched, serving as reference for future migration

After my departure, Jim:

1. **Deleted `legacy_lib/` entirely** and moved its contents to the root (`include/`, `src/`)
2. **Moved my `dft_lib/` to `mduran/dft_lib_refact/`** (a personal subfolder, effectively sidelining it)
3. **Moved my `Dockerfile` and `buildlib.sh` to `mduran/`** as well
4. **Continued developing** the legacy code in-place, adding ~466 commits (mostly "synch") with significant new physics

The repository now has a flat legacy structure at root level, with my clean modular code buried in `mduran/dft_lib_refact/`.

### My completed work (as of April 2021)

| Module | Headers | Source | Tests | Examples | Status |
|---|---|---|---|---|---|
| `utils/console` | `console.h` | (header-only) | 4 tests | console example | Complete |
| `utils/config_parser` | `config_parser.h` | `config_parser.cpp` | 7 tests | config_parser example | Complete |
| `utils/functions` | `functions.h` | (header-only) | 2 tests | — | Complete |
| `exceptions/grace` | `grace_exception.h` | `grace_exceptions.cpp` | 4 tests | — | Complete |
| `exceptions/parameter` | `parameter_exceptions.h` | `parameter_exceptions.cpp` | 2 tests | — | Complete |
| `graph/grace` | `grace.h` | `grace.cpp` | ~25 tests | graphs example | Complete |
| `numerics/arithmetic` | `arithmetic.h` | `arithmetic.cpp` | 10 tests | arithmetic example | Complete |
| `numerics/integration` | `integration.h` | (header-only, templates) | ~10 tests | integration example | Complete |
| `physics/potentials/intermolecular` | `potential.h` | `potential.cpp` | 7 tests | potentials example | Complete |
| `geometry/base` | `vertex.h`, `element.h`, `mesh.h` | `vertex.cpp`, `element.cpp`, `mesh.cpp` | 12 tests | geometry example | Complete |
| `geometry/2D` | `element.h`, `mesh.h` | `element.cpp`, `mesh.cpp` | 3 tests | (in geometry example) | Complete |
| `geometry/3D` | `element.h`, `mesh.h` | `element.cpp`, `mesh.cpp` | 0 tests | (in geometry example) | Partial (plot not implemented, no tests) |

**Total: 16 headers, 15 source files, ~86 tests, 7 example programs.**

### New physics Jim added (post-April 2021)

These are features that did not exist when you left and must be incorporated into the modern structure:

| Feature | Files | What it does |
|---|---|---|
| Crystal lattice builder | `Crystal_Lattice.h/cpp` | BCC/FCC/HCP crystal lattices with Miller-index orientations |
| Equations of state | `EOS.h` | LJ-JZG (32-param), Mecke, FMSA, PY hard-sphere EOS |
| DFT factory | `DFT_Factory.h` | End-to-end pipeline construction from input file |
| FMT helper (analytic weights) | `FMT_Helper.h` | Analytic G-functions for FMT weight computation on lattice |
| Dynamical matrix interface | `Dynamical_Matrix.h/cpp` | Abstract Hessian $\delta^2F/\delta\rho^2$ via matrix-vector products |
| Eigenvalue solver | `Eigenvalues.h/cpp` | Rayleigh quotient minimisation for smallest eigenvalue |
| Arnoldi eigensolver | `Arnoldi.h/cpp` | Implicitly Restarted Arnoldi for largest eigenvalues (optional, Armadillo) |
| Log-determinant estimator | `Log_Det.h/cpp` | Stochastic $\log\det(A)$ via Chebyshev polynomials |
| External fields | `External_Field.h`, `Density.h` | Arbitrary external potential $V_{\text{ext}}(\mathbf{r})$ |
| Spline interpolation | `Spliner.h/cpp` | 1D/2D cubic splines, smoothing splines |
| Alias coordinates | `Species.h`, `Minimizer.h`, `DFT.h` | Variable transforms $\rho = \rho_{\min} + x^2$ ensuring $\rho > 0$ |
| Saddle-point search | `Minimizer.h` | Fixed-direction projection for dimer/transition-state methods |
| Gaussian density init | `Density.h` | `initialize_with_gaussians()` for crystal initial conditions |
| Boundary system | `Lattice.h` | Configurable boundary width with point enumeration |

Key files that were **significantly modified** (not just moved):

- `DFT.h/cpp` — ~300 lines of additions (Hessian, alias support, external fields, EOS)
- `FMT.h/cpp` — ~400 lines (esFMT, second derivatives, tensor measures, lambda regulator)
- `Minimizer.h/cpp` — ~500 lines (DDFT, DDFT_IF, FIRE2 improvements, alias workflow)
- `Species.h` — ~600 lines (alias system, external field, EOS species, reflection)
- `Interaction.h/cpp` — ~360 lines (second derivatives, Gauss-Legendre, interpolation variants)
- `Lattice.h` — ~100 lines (boundary width system)
- `Density.h/cpp` — ~180 lines (VTK output, Gaussian init, background density)

---

## Architecture for the reinvented library

### Design principles

1. **C++20** (not C++14 or C++18). Use concepts, `std::span`, `std::format`, structured bindings, `constexpr` where possible, designated initialisers, `[[nodiscard]]`, three-way comparison.
2. **No raw `new`/`delete`**. Exclusive use of `std::unique_ptr`, `std::shared_ptr`, value semantics.
3. **Namespaces throughout**. Root namespace `dft_core`, sub-namespaces for each module.
4. **Header/source mirror layout**. `include/dft_lib/<module>/<file>.h` with `src/<module>/<file>.cpp`.
5. **100% GTest coverage**. Every public method, every edge case, every exception path.
6. **CMake modern practices**. Target-based dependency management, `target_link_libraries`, `FetchContent` for GTest, proper install targets.
7. **No `using namespace std` in headers**. Ever.
8. **No Boost dependency** for new code. Replace `boost::property_tree` with `nlohmann/json` or `toml++`. Replace `boost::serialization` with cereal or a custom solution. Replace `boost::combine` with C++20 `std::views::zip`.
9. **FFTW3 abstraction**. Wrap FFTW behind a clean C++ interface (RAII plans, span-based data access).
10. **Optional dependencies cleanly gated**. Grace (plotting), Armadillo, MPI, OpenMP as optional cmake options with feature-test macros.

### Coding philosophy

> The library must feel like a modern Python library to its users. When a client writes code with classicalDFT, the experience must be as readable, explicit, and Pythonic as working with scikit-learn or similar modern interfaces.

#### Naming conventions

- **No abbreviations in public class names.** The user instantiates `auto plan = FourierTransform(shape)` or `auto s = CompensatedSum()`. Short aliases (`auto ft = ...`) are the client's choice; the class name itself must read like prose.
- **Method names read as verbs or descriptors.** `find_hard_sphere_diameter()`, `compute_van_der_waals_integral()`, `derivative()`, `integrate()`. No abbreviated method names.
- **Namespaces are hierarchical and explicit.** `dft_core::numerics::fourier::FourierTransform` rather than `dft_core::fft::FFTPlan`.
- **Constants and enums use UPPER_SNAKE_CASE or PascalCase enum members** as appropriate.

#### Modern C++ (C++20) for a Pythonic feel

- **Concepts everywhere.** Replace SFINAE with `requires` clauses and named concepts. Type constraints read like Python type hints.
- **`std::span`** for non-owning views (like Python's memoryview / numpy slices).
- **Structured bindings** for multi-return values (like Python tuple unpacking).
- **`[[nodiscard]]`** on all accessors and pure functions (like Python linters warning on unused returns).
- **Designated initialisers** for option structs (like Python keyword arguments).
- **Three-way comparison (`<=>`)** for value types.
- **`constexpr`** for anything computable at compile time.
- **Range-based algorithms and views** where they improve readability.

#### Testing requirements

- **100% line coverage on every module.** No exceptions. Every public method, every error path, every branch.
- **Parameterized tests (pytest style).** Use GTest's `TEST_P` with `INSTANTIATE_TEST_SUITE_P` to test functions across multiple inputs, mirroring `@pytest.mark.parametrize`. Each function of a module gets its own parameterized test suite where applicable.
- **One test file per module.** Test file mirrors source file structure: `tests/numerics/fourier.cpp` tests `src/numerics/fourier.cpp`.
- **Test names describe behaviour, not implementation.** `InterpolateSinAtMidpoints`, not `test_spline_1`.

### Target directory structure

```
classicalDFT/
├── CMakeLists.txt
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
├── cmake/
│   ├── FindDependencies.cmake
│   └── CompilerWarnings.cmake
├── include/
│   └── dft_lib/
│       ├── dft_lib.h                          # umbrella header
│       ├── core/
│       │   ├── types.h                        # type aliases, constants
│       │   └── config.h                       # build configuration
│       ├── exceptions/
│       │   ├── grace_exception.h
│       │   ├── parameter_exception.h
│       │   └── dft_exception.h                # NEW: base exception
│       ├── utils/
│       │   ├── console.h
│       │   ├── config_parser.h
│       │   ├── functions.h
│       │   ├── table.h                        # NEW: from legacy Table.h
│       │   ├── timestamp.h                    # NEW: from legacy TimeStamp.h
│       │   └── log.h                          # NEW: from legacy Log.h
│       ├── graph/
│       │   └── grace.h
│       ├── numerics/
│       │   ├── arithmetic.h
│       │   ├── integration.h
│       │   ├── fourier.h                      # NEW: FFTW3 C++ wrapper
│       │   ├── linear_algebra.h               # NEW: from DFT_LinAlg.h
│       │   ├── spline.h                       # NEW: from Spliner.h
│       │   └── eigensolver.h                  # NEW: from Eigenvalues.h + Arnoldi.h
│       ├── geometry/
│       │   ├── base/
│       │   │   ├── vertex.h
│       │   │   ├── element.h
│       │   │   └── mesh.h
│       │   ├── 2D/
│       │   │   ├── element.h
│       │   │   └── mesh.h
│       │   ├── 3D/
│       │   │   ├── element.h
│       │   │   └── mesh.h
│       │   └── lattice.h                      # NEW: 3D periodic lattice (from Lattice.h)
│       ├── physics/
│       │   ├── potentials/
│       │   │   └── intermolecular/
│       │   │       └── potential.h
│       │   ├── thermodynamics/
│       │   │   ├── enskog.h                   # NEW: hard-sphere thermo
│       │   │   └── eos.h                      # NEW: equations of state
│       │   ├── crystal/
│       │   │   └── crystal_lattice.h          # NEW: BCC/FCC/HCP builder
│       │   ├── density/
│       │   │   ├── density.h                  # NEW: density profile
│       │   │   └── external_field.h           # NEW: external field
│       │   ├── species/
│       │   │   ├── species.h                  # NEW: species base
│       │   │   ├── fmt_species.h              # NEW: FMT species + weighted densities
│       │   │   └── fmt_species_eos.h          # NEW: EOS-corrected species
│       │   ├── fmt/
│       │   │   ├── fmt.h                      # NEW: FMT base + all models
│       │   │   ├── fundamental_measures.h     # NEW: 19-component measures struct
│       │   │   ├── weighted_density.h         # NEW: weighted density + FFT convolution
│       │   │   └── fmt_helper.h               # NEW: analytic G-functions
│       │   ├── interaction/
│       │   │   └── interaction.h              # NEW: mean-field interactions
│       │   └── dft/
│       │       ├── dft.h                      # NEW: core DFT solver
│       │       ├── dynamical_matrix.h         # NEW: Hessian interface
│       │       ├── dft_factory.h              # NEW: factory builder
│       │       └── coexistence.h              # NEW: phase coexistence
│       └── dynamics/
│           ├── minimizer.h                    # NEW: FIRE2 minimizer
│           ├── ddft.h                         # NEW: dynamical DFT
│           └── ddft_if.h                      # NEW: integrating factor DDFT
├── src/
│   ├── exceptions/
│   ├── utils/
│   ├── graph/
│   ├── numerics/
│   ├── geometry/
│   │   ├── base/
│   │   ├── 2D/
│   │   └── 3D/
│   └── physics/
│       ├── potentials/
│       ├── thermodynamics/
│       ├── crystal/
│       ├── density/
│       ├── species/
│       ├── fmt/
│       ├── interaction/
│       ├── dft/
│       └── dynamics/
├── tests/
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── exceptions/
│   ├── utils/
│   ├── graph/
│   ├── numerics/
│   ├── geometry/
│   └── physics/
│       ├── potentials/
│       ├── thermodynamics/
│       ├── crystal/
│       ├── density/
│       ├── species/
│       ├── fmt/
│       ├── interaction/
│       ├── dft/
│       └── dynamics/
├── examples/
│   ├── console/
│   ├── config_parser/
│   ├── graphs/
│   ├── geometry/
│   ├── numerics/
│   │   ├── arithmetic/
│   │   └── integration/
│   ├── physics/
│   │   ├── potentials/
│   │   ├── thermodynamics/
│   │   ├── crystal/
│   │   └── droplet/
│   └── dynamics/
│       └── fire_minimizer/
└── docs/
    ├── installation/
    └── latex-notes/
```

---

## Phased execution plan

### Phase 0: Repository setup (fork and scaffold)

**Goal:** Fork the repository under my account, restore my code as the canonical structure, set up CI.

| Step | Task | Details |
|---|---|---|
| 0.1 | Fork `jimlutsko/classicalDFT` to my GitHub account | `gh repo fork jimlutsko/classicalDFT --clone=false` |
| 0.2 | Clone my fork locally | Fresh working copy |
| 0.3 | Create `main` branch from my last clean commit (`9747248`) | `git checkout -b main 9747248` — this becomes the starting point |
| 0.4 | Restructure: move `dft_lib/` contents to root-level `include/dft_lib/`, `src/`, `tests/`, `examples/` | Follow the target directory structure above |
| 0.5 | Modernise CMakeLists.txt | C++20, `FetchContent` for GTest, target-based deps, install targets |
| 0.6 | Add `.github/workflows/ci.yml` | Build + test on Ubuntu, macOS; GCC 12+, Clang 15+ |
| 0.7 | Add `Dockerfile` updated for modern toolchain | Ubuntu 22.04+, GCC 12, CMake 3.25+, FFTW3, GSL |
| 0.8 | Verify all existing 86 tests pass | `cmake --build build && ctest --test-dir build` |
| 0.9 | Tag `v2.0.0-alpha.1` | First checkpoint |

**Estimated scope:** ~half a day of work. Zero new code, pure restructure.

---

### Phase 1: Modernise existing modules to C++20

**Goal:** Upgrade my already-complete code from C++14 to C++20, remove Boost dependencies where possible, tighten the API.

| Step | Task | Details |
|---|---|---|
| 1.1 | `utils/console.h` — move into `dft_core` namespace | Currently in bare `console::` namespace; wrap in `dft_core::utils::console` |
| 1.2 | `utils/config_parser.h` — replace Boost.PropertyTree | Replace with `nlohmann/json` (JSON), `toml++` (TOML/INI), or keep Boost as optional. Add YAML support |
| 1.3 | `utils/functions.h` — replace SFINAE with C++20 concepts | `template<std::floating_point T>` instead of `std::enable_if` |
| 1.4 | `graph/grace.h` — add `[[nodiscard]]`, `std::string_view` parameters | Minor modernisation. Keep xmgrace dependency as optional |
| 1.5 | `numerics/arithmetic.h` — replace SFINAE with concepts, use `std::span` | `std::span<const double>` instead of `const std::vector<double>&` |
| 1.6 | `numerics/integration.h` — replace `std::function` with concepts | `template<typename F> requires std::invocable<F, double>` |
| 1.7 | `physics/potentials/intermolecular/potential.h` — remove Boost.Serialization | Replace with cereal or custom binary serialization. Add `[[nodiscard]]` |
| 1.8 | `geometry/*` — add concepts for vertex/element/mesh constraints | `concept Dimensional`, `concept MeshLike`. Use `std::span` |
| 1.9 | `exceptions/*` — add `[[nodiscard]]` to `what()`, use `std::source_location` | |
| 1.10 | Update all tests for any API changes | Ensure 86+ tests still pass |
| 1.11 | Tag `v2.0.0-alpha.2` | |

**Estimated scope:** 1-2 days. Mechanical upgrades, no new physics.

---

### Phase 2: Numerics foundation

**Goal:** Build the low-level numerical infrastructure that all physics modules depend on, leveraging existing libraries (Armadillo, GSL, FFTW3) rather than reimplementing.

#### Design decisions

1. **No custom vector types.** Jim's `DFT_Vec`/`DFT_Vec_Complex` are thin wrappers around `fftw_malloc`. Armadillo's `arma::vec`/`arma::cx_vec` already provide all the same operations (dot, norm, scale, element access) with BLAS/LAPACK acceleration. We use `arma::vec` directly everywhere linear algebra is needed.

2. **`GridShape` value type.** Jim's code scatters `Nx_`, `Ny_`, `Nz_` as separate members in every class that touches the 3D grid (`DFT_FFT`, `Lattice`, `Arnoldi`, `Species`, etc.). We introduce a single `struct GridShape{nx, ny, nz}` that encapsulates grid dimensions. `UniformMesh` provides a `GridShape` to its dependents via `grid_shape()`.

3. **FourierTransform: thin RAII wrapper only.** `FourierTransform` owns FFTW-aligned buffers and `fftw_plan` handles. It exposes data via `std::span<double>` and `std::span<std::complex<double>>` for zero-copy access. Users that need `arma::vec` operations wrap trivially: `arma::vec v(plan.real().data(), plan.real().size(), false, true)`.

4. **Splines via GSL.** Jim's `Spliner` is a Numerical Recipes cubic spline reimplementation with raw `double*` arrays. GSL (already a dependency for integration) provides `gsl_spline` (1D) and `gsl_spline2d` (2D bicubic) with evaluation, derivatives, and integration. We wrap these with RAII, not reimplement the algorithm.

5. **No standalone eigensolver.** Dense eigenproblems: use `arma::eig_sym(eigval, eigvec, A)` directly. Matrix-free Hessian (the real DFT use case): Jim's `Arnoldi.cpp` already uses Armadillo internally (`arma::cx_vec`, `arma::eig_gen`, `arma::qr`). This is a Phase 9 concern (dynamics) where the `DynamicalMatrix` interface exists. Providing premature power-iteration solvers adds code with no downstream consumer.

| Step | Task | Source | Details |
|---|---|---|---|
| 2.1 | `numerics/grid_shape.h` — grid dimension value type | NEW | `struct GridShape{unsigned nx, ny, nz}` with `total()`, `fourier_total()`, three-way comparison. Used by `FourierTransform`, `FourierConvolution`, and downstream geometry/physics code. Avoids scattering raw dimension triples |
| 2.2 | `numerics/fourier.h` + `src/numerics/fourier.cpp` — FFTW3 RAII wrapper | `DFT_LinAlg.h` | `FourierTransform`: move-only, `GridShape`-based construction, owns `fftw_malloc`'d buffers, `std::span` access, `forward()`/`backward()`. `FourierConvolution`: composes three `FourierTransform` objects for cyclic convolution $c = \text{IFFT}[\hat{a} \cdot \hat{b}] / N$ |
| 2.3 | Tests for `fft.h` | — | Round-trip accuracy (forward+backward recovers input), Parseval's theorem, convolution of known signals |
| 2.4 | `numerics/spline.h` + `src/numerics/spline.cpp` — GSL-backed splines | `Spliner.h` | `CubicSpline`: RAII wrapper around `gsl_spline`/`gsl_interp_accel`, supports eval/derivative/derivative2/integrate via `std::span` input. `BivariateSpline`: wraps `gsl_spline2d` for 2D regular-grid bicubic interpolation with partial derivatives |
| 2.5 | Tests for `spline.h` | — | Interpolation accuracy on known functions ($\sin x$, $x^3$), derivative correctness, integral verification against analytical values |
| 2.6 | Tag `v2.0.0-alpha.3` | | |

**What was removed vs the original plan:** `linear_algebra.h` (custom `RealVector`/`ComplexVector` — use `arma::vec`), `eigensolver.h` (custom Rayleigh quotient / inverse iteration — use `arma::eig_sym` or defer Arnoldi to Phase 9).

**Estimated scope:** 1-2 days. Thin wrappers over proven libraries.

---

### Phase 3: Geometry — uniform mesh with periodic boundaries

**Goal:** Extend the existing geometry module with `UniformMesh`, a structured uniform mesh that adds periodic boundary conditions and FFT interoperability. This replaces the legacy "Lattice" concept: there is no separate lattice class. The mesh **is** the computational domain.

#### Design decisions

1. **Build on the existing hierarchy, do not bypass it.** `UniformMesh` inherits from `SUQMesh`, which inherits from `Mesh`. It stores real `Vertex` and `Element` objects explicitly (not computed on-the-fly). Explicit is better than implicit. This is the same design used by production mesh engines (deal.II, FEniCS, MFEM) and keeps the door open for future FEM-based DFT.

2. **Both 2D and 3D.** `two_dimensional::UniformMesh` extends `two_dimensional::SUQMesh`. `three_dimensional::UniformMesh` extends `three_dimensional::SUQMesh`. The physics code should be general enough to work in either dimension.

3. **No separate `PeriodicLattice`.** Jim's old `Lattice` class conflated grid shape, physical spacing, index conversion, and PBC into a monolith disconnected from geometry. We do not replicate it. `UniformMesh` is a proper `Mesh` subclass that happens to also provide PBC and a `GridShape` accessor for the Fourier layer.

4. **`GridShape` stays a lightweight value type.** `FourierTransform` keeps taking `GridShape` (decoupled from geometry). `UniformMesh` provides it via `.grid_shape()`. This keeps the Fourier module testable without constructing a full mesh.

5. **Spacing per axis.** The base `SUQMesh` uses a single `dx` for all directions (square cells). `UniformMesh` adds per-axis spacing (`dx`, `dy`, and `dz` in 3D; `dx`, `dy` in 2D) to support rectangular (non-square) cells. The underlying elements are still `SquareBox` (the cell shape is the same, but sizes can differ per direction in future refinements). For now, `UniformMesh` constructs with a single `dx` (uniform) and exposes the spacing.

#### What `UniformMesh` adds over `SUQMesh`

| Feature | Method | Description |
|---|---|---|
| Grid shape | `grid_shape()` | Returns `GridShape{nx, ny, nz}` (or `{nx, ny, 1}` for 2D) for FFT interop |
| Spacing | `spacing()` | Returns `{dx, dy, dz}` or `{dx, dy}` as `std::vector<double>` |
| Box size | `box_size()` | `dimensions()` (inherited, same thing) |
| PBC wrap | `wrap(position)` | Wraps a spatial position into the periodic box: $x_i \mapsto x_i \bmod L_i$ |
| Cell volume | `cell_volume()` | Product of spacings (equivalent to `element_volume()`, aliased) |
| Position from index | `position(flat_index)` | Returns the spatial coordinates of a grid point by its global index |

| Step | Task | Details |
|---|---|---|
| 3.1 | `geometry/2D/uniform_mesh.h` + `.cpp` | `two_dimensional::UniformMesh : public two_dimensional::SUQMesh`. Constructor takes `(dx, dimensions, origin)`. Adds `grid_shape()`, `spacing()`, `wrap()`, `position()`, `cell_volume()` |
| 3.2 | `geometry/3D/uniform_mesh.h` + `.cpp` | `three_dimensional::UniformMesh : public three_dimensional::SUQMesh`. Same interface |
| 3.3 | Tests for `UniformMesh` (2D and 3D) | Construction, `grid_shape()` matches shape, PBC wrap, position round-trip, cell volume |
| 3.4 | Update umbrella header | Add 2D and 3D `uniform_mesh.h` to `<classicaldft>` |
| 3.5 | Tag `v2.0.0-alpha.4` | |

**Estimated scope:** 1-2 days.

---

### Phase 4: Thermodynamics layer

**Goal:** Migrate the equation-of-state and hard-sphere thermodynamics code.

| Step | Task | Source | Details |
|---|---|---|---|
| 4.1 | `physics/thermodynamics/enskog.h` | `Enskog.h` | `namespace dft_core::physics::thermodynamics`. Free functions for PY/Carnahan-Starling: `eta_from_density(rho, hsd)`, `free_energy_cs(eta)`, `pressure_cs(eta)`, `chemical_potential_cs(eta)`, `contact_value(eta)`. Plus transport coefficients (`viscosity`, `diffusion`, `thermal_conductivity`) |
| 4.2 | Tests for `enskog.h` | — | Known limiting values ($\eta \to 0$, known CS values at $\eta = 0.49$), virial expansion coefficients |
| 4.3 | `physics/thermodynamics/eos.h` | `EOS.h` | Abstract `class EOS` with `phi_ex(rho)`, `dphi_ex(rho)`, `d2phi_ex(rho)`, `d3phi_ex(rho)`, `free_energy_per_atom(rho)`, `pressure(rho)`. Concrete: `NullEOS`, `PercusYevickEOS`, `JohnsonZollwegGubbinsEOS`, `MeckeEOS`, `FMSA_EOS`. Store parameters as `constexpr` arrays where possible |
| 4.4 | Tests for `eos.h` | — | Thermodynamic identities ($p = \rho^2 \partial f/\partial \rho$), known tabulated values, consistency between derivatives |
| 4.5 | `physics/crystal/crystal_lattice.h` | `Crystal_Lattice.h/cpp` | `class CrystalLattice` with `enum class Structure { BCC, FCC, HCP }`, `enum class Orientation { _001, _010, _100, _110, _101, _011, _111 }`. Clean constructor, `positions()` returns `std::vector<std::array<double,3>>`. XYZ export |
| 4.6 | Tests for `crystal_lattice.h` | — | Coordination numbers, nearest-neighbor distances, symmetry checks |
| 4.7 | Tag `v2.0.0-alpha.5` | | |

**Estimated scope:** 2-3 days.

---

### Phase 5: Density and species

**Goal:** Migrate the density profile and species management — the data layer of DFT.

| Step | Task | Source | Details |
|---|---|---|---|
| 5.1 | `physics/density/density.h` | `Density.h/cpp` | `class Density` owns a `UniformMesh` and an `arma::vec` for the density profile. FFT via `FourierTransform`. Methods: `set`, `get`, `doFFT`, `getNumberAtoms`, `getCenterOfMass`, `writeVTK`, `initializeWithGaussians`, `expand`, `resize`. No raw arrays: use `operator[]` with bounds checking in debug builds |
| 5.2 | `physics/density/external_field.h` | `Density.h`, `External_Field.h` | `class ExternalField` holds an `arma::vec` and species index. Serializable. Simple container |
| 5.3 | Tests for density and external field | — | FFT round-trip, atom counting ($\sum\rho\,dV = N$), VTK output, Gaussian initialization |
| 5.4 | `physics/species/species.h` | `Species.h` | `class Species`: owns `Density`, force vector, chemical potential. Constraint system (fixed mass, fixed background, homogeneous boundary). Alias coordinates ($\rho = \rho_{\min} + x^2$). Force management (zero, add, begin/end calculation). Reflection symmetry |
| 5.5 | Tests for species | — | Alias round-trip ($x \to \rho \to x$), constraint enforcement, force accumulation |
| 5.6 | Tag `v2.0.0-alpha.6` | | |

**Estimated scope:** 2-3 days.

---

### Phase 6: Fundamental Measure Theory

**Goal:** Migrate the FMT engine — the most mathematically dense part of the library.

| Step | Task | Source | Details |
|---|---|---|---|
| 6.1 | `physics/fmt/fundamental_measures.h` | `Fundamental_Measures.h` | `struct FundamentalMeasures`: 19 components ($\eta, s_0, s_1, s_2, \mathbf{v}_1, \mathbf{v}_2, \mathbf{T}$) plus derived quantities (`vTv`, `Tr_T2`, `Tr_T3`). Use `std::array` for components. `calculate_derived()` method |
| 6.2 | `physics/fmt/weighted_density.h` | `FMT_Weighted_Density.h` | `class WeightedDensity`: owns weight array, weighted density, $d\Phi$ array, all as `arma::vec`. Fourier convolution via `FourierConvolution`. RAII for all arrays |
| 6.3 | `physics/fmt/fmt_helper.h` | `FMT_Helper.h` | `class FMTHelper`: static methods for analytic G-functions (`G_eta`, `G_s`, `G_vx`, `G_txx`, `G_txy`, `G_I1x`, `G_I2`). Document the underlying integrals with LaTeX in doc-strings. Add NaN-guard assertions |
| 6.4 | `physics/fmt/fmt.h` | `FMT.h/cpp` | Abstract `class FMT`: `calculateFreeEnergyAndDerivatives()`, `addSecondDerivative()`. Protected pure virtuals for `f1_`, `f2_`, `f3_` and their derivatives. Concrete via inheritance: `class Rosenfeld`, `class RSLT`, `class ExplicitlyStableFMT`, `class WhiteBearI`, `class WhiteBearII`. Each model defines $\Phi_3$ and its cross-derivatives |
| 6.5 | `physics/species/fmt_species.h` | `Species.h`, `FMT_Species.cpp` | `class FMTSpecies : public Species`: 11-component weighted densities, `Initialize()` (generates weight functions), `calculateFundamentalMeasures()` (FFT convolution), `calculateForce()`. Alias system for $\eta < 1$ |
| 6.6 | `physics/species/fmt_species_eos.h` | `FMT_Species_EOS.cpp` | `class FMTSpeciesEOS : public FMTSpecies`: EOS correction $\Delta F_{\text{EOS}}$ |
| 6.7 | Tests for FMT | — | Known Rosenfeld free energy at specific $\eta$ values, CS pressure recovery, consistency between models, weight function symmetry, FFT convolution accuracy |
| 6.8 | Tag `v2.0.0-beta.1` | | |

**Estimated scope:** 3-5 days. This is the hardest phase.

---

### Phase 7: Interactions

**Goal:** Migrate the mean-field interaction hierarchy.

| Step | Task | Source | Details |
|---|---|---|---|
| 7.1 | `physics/interaction/interaction.h` | `Interaction.h/cpp` | Abstract `class Interaction`: weight array $w(\mathbf{R})$, FFT convolution, `getInteractionEnergyAndForces()`, `addSecondDerivative()`. Concrete hierarchy: `class GaussInteraction` (energy/force routes), `class InterpolationInteraction` (zero/linear/quadratic x energy/force). Use `std::variant` or template parameters instead of deep inheritance |
| 7.2 | Tests for interactions | — | Energy conservation, force-energy consistency ($F = -\nabla E$), symmetry of weight array, known limiting cases (ideal gas) |
| 7.3 | Tag `v2.0.0-beta.2` | | |

**Estimated scope:** 1-2 days.

---

### Phase 8: Core DFT solver

**Goal:** Migrate the central DFT class that ties everything together.

| Step | Task | Source | Details |
|---|---|---|---|
| 8.1 | `physics/dft/dynamical_matrix.h` | `Dynamical_Matrix.h/cpp` | Abstract `class DynamicalMatrix`: interface for $H \cdot v$ products. Boundary masking, squared-matrix mode, alias-coordinate support. Clean virtual interface |
| 8.2 | `physics/dft/dft.h` | `DFT.h/cpp` | `class DFT : public DynamicalMatrix`: the core orchestrator. Owns `vector<unique_ptr<Species>>`, `vector<unique_ptr<Interaction>>`, `unique_ptr<FMT>`, `vector<ExternalField>`. Core method: `calculateFreeEnergyAndDerivatives()` decomposing $\Omega[\rho]$ into ideal + hard-sphere + EOS + mean-field + external. Bulk thermo: `chemicalPotential()`, `grandPotential()`, `helmholtzFreeEnergy()`, `findSpinodal()`, `findCoexistence()`, `getCriticalPoint()`. Structural: `realSpaceDCF()`, `fourierSpaceDCF()`. Hessian: `matrixDotV()`, `diagonalElements()` |
| 8.3 | `physics/dft/coexistence.h` | `DFT_Coex.cpp` | Free functions for vapour-liquid coexistence: `findCoexistence(eos, kT)`, `findSpinodal(eos, kT)`, `getCriticalPoint(eos)` |
| 8.4 | `physics/dft/dft_factory.h` | `DFT_Factory.h` | `class DFTBuilder` (builder pattern, not factory): constructs DFT from configuration. Uses `std::unique_ptr` throughout, RAII ownership. Replaces raw `new` with `make_unique` |
| 8.5 | Tests for DFT | — | Ideal gas free energy ($F = \int\rho\ln\rho\,dV$), known coexistence points, force consistency (numerical gradient vs analytic), Hessian symmetry |
| 8.6 | Tag `v2.0.0-beta.3` | | |

**Estimated scope:** 3-4 days.

---

### Phase 9: Dynamics (minimiser + DDFT)

**Goal:** Migrate the time-evolution and minimisation algorithms.

| Step | Task | Source | Details |
|---|---|---|---|
| 9.1 | `dynamics/minimizer.h` | `Minimizer.h/cpp` | Abstract `class Minimizer`: `run()`, `resume()`, `step()`. Concrete `class FIRE2Minimizer`: velocity-Verlet with FIRE2 adaptive timestep. Convergence monitoring, alias-coordinate workflow, fixed-direction projection for saddle points. Backtracking on $\eta$ overflow |
| 9.2 | `dynamics/ddft.h` | `Minimizer.h` (DDFT class) | `class DDFT : public Minimizer`: finite-difference diffusion + implicit excess forces. `enum class DiffType { Forward1, Central, Forward2, Forward3 }`. Adaptive timestep |
| 9.3 | `dynamics/ddft_if.h` | `Minimizer.h` (DDFT_IF class) | `class DDFT_IF : public DDFT`: integrating-factor method with exact diffusion propagator via FFT. Crank-Nicholson implicit. Optional Arnoldi-based eigenvector detection |
| 9.4 | Tests for dynamics | — | FIRE2 convergence on quadratic potential, DDFT diffusion accuracy (compare with analytic solution for free diffusion), energy decrease monotonicity |
| 9.5 | Example: droplet nucleation | `examples/Droplet/` | Port legacy droplet example to modern API |
| 9.6 | Tag `v2.0.0-beta.4` | | |

**Estimated scope:** 2-3 days.

---

### Phase 10: Utilities, documentation, and polish

**Goal:** Migrate remaining utilities, write documentation, finalize.

| Step | Task | Details |
|---|---|---|
| 10.1 | `utils/table.h` | From `Table.h`: simple file reader. Modernize with `std::filesystem` |
| 10.2 | `utils/log.h` | From `Log.h`: logging to file. Integrate with `console.h` |
| 10.3 | `utils/timestamp.h` | From `TimeStamp.h`: chrono-based timing |
| 10.4 | Replace Grace with optional modern plotting | Consider adding a thin gnuplot or matplotlib-cpp backend as alternative to xmgrace |
| 10.5 | Write comprehensive README.md | Installation, build, usage, examples, API overview |
| 10.6 | Write Doxygen configuration | Generate API docs from doc-strings |
| 10.7 | Write mathematical documentation | LaTeX notes for FMT, DDFT, potentials (extend existing `documentation/latex-notes/`) |
| 10.8 | Final test coverage audit | Ensure every public method has tests. Target: 100% line coverage on physics code |
| 10.9 | Performance benchmarks | Compare with legacy library on standard problems (droplet, coexistence) |
| 10.10 | Tag `v2.0.0` | |

**Estimated scope:** 2-3 days.

---

## Dependency map

The migration order above respects these dependencies (each module depends only on modules from earlier phases):

```
Phase 0-1: utils, exceptions, graph (no physics deps)
     │
     v
Phase 2: numerics/grid_shape, numerics/fourier (FFTW3 wrapper), numerics/spline (GSL wrapper)
         ── arma::vec used directly for all linear algebra, arma::eig_sym for dense eigenproblems
     │
     v
Phase 3: geometry/lattice (owns GridShape, provides it to FFT layer)
     │
     v
Phase 4: thermodynamics/enskog, thermodynamics/eos, crystal (depends on lattice, potentials)
     │
     v
Phase 5: density, species (depends on lattice, arma::vec, FFTPlan)
     │
     v
Phase 6: FMT (depends on density, species, FFTConvolution)
     │
     v
Phase 7: interaction (depends on density, FFTConvolution, potentials)
     │
     v
Phase 8: DFT solver (depends on FMT, interaction, species, density)
     │
     v
Phase 9: dynamics (depends on DFT, arma::vec, Arnoldi eigensolver using arma::cx_vec/eig_gen/qr)
     │
     v
Phase 10: polish, docs, benchmarks
```

---

## Key decisions to make before starting

1. **Serialization**: Boost.Serialization is deeply embedded in the legacy code (Density, Species, Lattice, Potential, Minimizer, Eigenvalues all use it). Options:
   - **cereal** (header-only, modern C++, similar API) — recommended
   - **protobuf** (cross-language, schema-based) — overkill for this
   - Custom binary I/O — fragile

2. **Armadillo dependency**: Used in `potential.h` (`arma::vec` overloads), will be used as the primary vector/matrix library throughout. `arma::vec` replaces all custom vector types. `arma::eig_sym`/`arma::eig_gen` handle dense eigenproblems. `arma::cx_vec`/`arma::cx_mat`/`arma::qr` power the Arnoldi solver. **Decision: required dependency, not optional.**

3. **Grace dependency**: xmgrace is old and platform-dependent. Options:
   - Keep as optional backend behind cmake option
   - Add gnuplot backend as alternative
   - Add file-only output (CSV/VTK) as default, plotting as optional extra

4. **MPI/OpenMP**: The legacy code has optional MPI/OpenMP support. Options:
   - Keep as cmake options (`DFT_USE_MPI`, `DFT_USE_OMP`)
   - Gate behind abstract execution policy (future std::execution integration)

5. **Config file format**: Boost.PropertyTree supports INI/JSON/XML/INFO. Options:
   - Switch to TOML (via `toml++`, header-only) as primary format
   - Keep JSON via `nlohmann/json` (header-only)
   - Support multiple formats via adapter pattern

---

## Files to recover from git history

My code exists intact at commit `9747248`. Key recovery commands:

```bash
# Recover my complete dft_lib/
git show 9747248:dft_lib/core/classical_dft > ...

# Or bulk checkout
git checkout 9747248 -- dft_lib/
```

Jim's legacy code at the point of my last commit can be recovered from:

```bash
# Recover legacy_lib/ as it was when you left
git checkout 9747248 -- legacy_lib/
```

The new code Jim wrote (post-April 2021) can be extracted from `HEAD`:

```bash
# New files only
git diff --diff-filter=A --name-only 9747248..HEAD | grep -v mduran/
```

---

## Verification checklist

For each phase, before tagging a release:

- [ ] All tests pass (`ctest --test-dir build -V`)
- [ ] No compiler warnings (`-Wall -Wextra -Wpedantic -Werror`)
- [ ] No memory leaks (run with AddressSanitizer: `-fsanitize=address`)
- [ ] No undefined behaviour (run with UBSan: `-fsanitize=undefined`)
- [ ] Doxygen generates clean docs (no warnings)
- [ ] CI passes on Ubuntu (GCC 12+) and macOS (Clang 15+)
- [ ] No `using namespace std` in any header
- [ ] No raw `new`/`delete` in new code
- [ ] Every public method has at least one test
- [ ] Examples compile and produce expected output
