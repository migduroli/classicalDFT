# classicalDFT renovation plan

This document is the complete architectural plan for transforming classicalDFT
from a 90s-style OOP library into a modern C++20 value-oriented, pure-header
library. Every decision made during the planning phase is recorded here so
nothing is lost.

---

## 1. Motivation

The library was originally written using deeply nested
inheritance hierarchies, mutable God objects, protected state, and procedural
orchestration patterns. A first round of refactoring reorganised the code and
improved readability, but inherited much of the original design.

The goal of this renovation is to rewrite the library following modern C++20
data-oriented design principles, inspired by Pydantic and Flama: data is
strictly separated from logic, all physics and algorithms are expressed as pure
functions, and the API is declarative, minimal, and explicit.

---

## 2. Design rules

These rules apply to every public symbol in the library.

### 2.1. Data is public, logic is free functions

Data structures are `struct` with public members. No getters, no setters. If a
value is just data, it is public. Logic lives in free functions that receive
data by const-reference and return results by value.

### 2.2. Nouns for data, verbs for free functions

- Bad: `class Solver { void compute_free_energy(); }`
- Good: `struct State { ... };` and `auto free_energy(const State&, const Model&) -> double;`

### 2.3. No prefixes on functions

| Banned prefix | Replacement |
|---------------|-------------|
| `compute_` | Direct verb: `free_energy()`, `forces()` |
| `get_` | Public field access: `state.temperature` |
| `set_` | Public field assignment: `state.temperature = 1.0` |
| `apply_` | Direct verb: `forces()`, `constrain_mass()` |
| `create_` on data | Aggregate init: `Grid{.dx = 0.1, ...}` |
| `make_` | Allowed only for validated factory functions |

### 2.4. Immutability and value semantics

Functions do not mutate inputs. They return new values. RVO and `std::move`
make this zero-copy. Algorithm functions take `State` by value so they own
their local working copy and return the final state.

### 2.5. `std::variant` over inheritance

No `virtual` keyword anywhere in the core library. Runtime polymorphism is
handled via `std::variant` + `std::visit`, which is cache-friendly and allows
aggressive compiler inlining.

### 2.6. `[[nodiscard]]` everywhere

Every function that returns a value is marked `[[nodiscard]]`.

### 2.7 Designated initializers for all struct construction

All structs are constructed using designated initializers for maximum
readability:

```cpp
physics::Model model{
    .grid = core::make_grid(0.1, {10.0, 10.0, 10.0}),
    .species = { physics::Species{.name = "Argon", .hard_sphere_diameter = 1.0} },
};
```

---

## 3. New directory structure

```
include/dft/
  core/
    grid.hpp                    # lightweight Grid for DFT pipelines
    density.hpp                 # pure data struct
    species.hpp                 # Species (identity) + SpeciesState (mutable)
    state.hpp                   # State aggregate
  math/
    fourier.hpp                 # FFTW3 RAII wrapper (kept, already modern)
    convolution.hpp             # extracted from ConvolutionField
    arithmetic.hpp              # compensated summation (kept)
    spline.hpp                  # cubic spline GSL wrapper (kept)
    integration.hpp             # numerical integration (kept)
    autodiff.hpp                # autodiff adapter (kept)
    hessian.hpp                 # modernized (concept or type-erased callable)
  physics/
    potentials.hpp              # variant-based potentials
    interactions.hpp            # Interaction spec struct
    fmt/
      models.hpp                # FMT model structs + free functions
      measures.hpp              # Measures struct (kept, already good)
      weights.hpp               # FmtWeightSet + generate()
    model.hpp                   # Model aggregate (replaces Solver)
  functionals/
    ideal_gas.hpp               # ideal gas contribution
    hard_sphere.hpp             # FMT hard sphere contribution
    mean_field.hpp              # mean-field interaction contribution
    external_field.hpp          # external field contribution
    functionals.hpp             # orchestrator returning Result
    bulk/
      thermodynamics.hpp        # pressure, chemical_potential, grand_potential
      phase_diagram.hpp         # continuation-based coexistence/spinodal
  algorithms/
    solvers/
      newton.hpp                # generic Newton-Raphson (C++20 concepts)
      continuation.hpp          # pseudo-arclength continuation
      jacobian.hpp              # finite-difference Jacobian
    alias.hpp                   # variant-based alias transforms
    fire.hpp                    # FIRE2 minimizer as free function
    split_operator.hpp          # DDFT split-operator integration
    crank_nicholson.hpp         # DDFT Crank-Nicholson integration
  geometry/
    vertex.hpp                  # value type (no virtual)
    element.hpp                 # value type + variant (2D/3D)
    mesh.hpp                    # value type + variant (2D/3D)
  crystal/
    lattice.hpp                 # minor cleanup
    types.hpp                   # kept as-is
  config/
    parser.hpp                  # kept, minor cleanup
  init.hpp                      # convenience state factories
  console.hpp                   # kept as-is (utility)
  plotting/
    grace.hpp                   # kept, decoupled from core
    exceptions.hpp              # kept
  exceptions.hpp                # kept (math + general)
  dft.hpp                       # umbrella header
```

---

## 4. Naming conventions

### 4.1. Type names

| Planned name | What it replaces | Kind |
|---|---|---|
| `Grid` | `Density::dx_`, `box_size_`, `shape_` (extracted) | `struct` |
| `Density` | `dft::density::Density` | `struct` (data only, no methods) |
| `Species` | `dft::species::Species` (identity part) | `struct` |
| `SpeciesState` | `dft::species::Species` (mutable part) | `struct` |
| `State` | new | `struct` (species states + temperature) |
| `Model` | `dft::Solver` | `struct` (grid + species + interactions + fmt) |
| `Interaction` | `dft::functional::interaction::Interaction` | `struct` (spec only) |
| `LennardJones` | `dft::potentials::LennardJones` | `struct` (parameters only) |
| `TenWoldeFrenkel` | `dft::potentials::tenWoldeFrenkel` | `struct` |
| `WangRamirezDobnikarFrenkel` | `dft::potentials::WangRamirezDobnikarFrenkel` | `struct` |
| `Potential` | `dft::potentials::Potential` (virtual hierarchy) | `std::variant<...>` |
| `SplitScheme` | bh_perturbation_ + r_attractive_min_ flags | `struct` |
| `Rosenfeld`, `RSLT`, `WhiteBearI`, `WhiteBearII` | same (already structs) | kept |
| `FMTModel` | same (already a variant) | kept |
| `FmtWeightSet` | `dft::functional::fmt::WeightSet` | `struct` holding `FourierTransform`s |
| `Measures` | same | kept as-is |
| `Contribution` | new | `struct` (free_energy + forces for one term) |
| `Result` | new | `struct` (full evaluation: all energies + forces) |
| `Solution` | new | `struct` (state + evaluation + steps + converged) |
| `FireConfig` | `dft::dynamics::Fire2Config` | `struct` |
| `SplitOperatorConfig` | `dft::dynamics::IntegratorConfig` (SO part) | `struct` |
| `CrankNicholsonConfig` | `dft::dynamics::IntegratorConfig` (CN part) | `struct` |
| `UnboundedAlias`, `BoundedAlias` | virtual `set_density_from_alias()` | `struct`s in variant |
| `AliasTransform` | `Species` virtual alias methods | `std::variant<...>` |
| `NewtonConfig` | Numerix `SolverKernel` settings | `struct` |
| `SolverResult` | Numerix `SolverKernel::Solve()` return | `struct` |
| `CurvePoint` | Numerix `MoorePenrose` state | `struct` |
| `ContinuationConfig` | Numerix `MoorePenrose` settings | `struct` |
| `PhasePoint` | new | `struct` (rho_vapor + rho_liquid + temperature) |
| `SquareBox2D`, `SquareBox3D` | deep `SquareBox` hierarchy | `struct`s in variant |
| `UniformMesh2D`, `UniformMesh3D` | deep `UniformMesh` hierarchy | `struct`s in variant |
| `Mesh` | `dft::geometry::Mesh` (virtual) | `std::variant<...>` |

### 4.2. Function names

#### `dft::core`

| Function | Signature |
|---|---|
| `make_grid` | `auto make_grid(double dx, std::array<double,3> box) -> Grid` |
| `atom_count` | `auto atom_count(const Grid&, const Density&) -> double` |
| `center_of_mass` | `auto center_of_mass(const Grid&, const Density&) -> arma::rowvec3` |

#### `dft::init`

| Function | Signature |
|---|---|
| `homogeneous` | `auto homogeneous(const Model&, double density) -> State` |
| `from_profile` | `auto from_profile(const Model&, const arma::vec& rho) -> State` |
| `from_file` | `auto from_file(const Model&, const std::string& path) -> State` |

#### `dft::physics::potentials`

| Function | Signature |
|---|---|
| `energy` | `auto energy(const Potential&, double r) -> double` |
| `energy_r2` | `auto energy_r2(const Potential&, double r2) -> double` |
| `attractive` | `auto attractive(const Potential&, double r, const SplitScheme&) -> double` |
| `repulsive` | `auto repulsive(const Potential&, double r, const SplitScheme&) -> double` |
| `hard_sphere_diameter` | `auto hard_sphere_diameter(const Potential&, double kT) -> double` |
| `vdw_integral` | `auto vdw_integral(const Potential&, double kT) -> double` |
| `r_min` | `auto r_min(const Potential&) -> double` |
| `hard_core_diameter` | `auto hard_core_diameter(const Potential&) -> double` |

#### `dft::physics::fmt`

| Function | Signature |
|---|---|
| `free_energy_density` | `auto free_energy_density(const FMTModel&, const Measures&) -> double` |
| `derivatives` | `auto derivatives(const FMTModel&, const Measures&) -> Measures` |
| `bulk_free_energy` | `auto bulk_free_energy(const FMTModel&, double rho, double d) -> double` |
| `excess_chemical_potential` | `auto excess_chemical_potential(const FMTModel&, double rho, double d) -> double` |
| `requires_tensor` | `auto requires_tensor(const FMTModel&) -> bool` |
| `name` | `auto name(const FMTModel&) -> std::string` |
| `weights` | `auto weights(double diameter, const Grid&) -> FmtWeightSet` |

#### `dft::functionals`

| Function | Signature |
|---|---|
| `ideal_gas` | `auto ideal_gas(const Grid&, const State&) -> Contribution` |
| `hard_sphere` | `auto hard_sphere(const Grid&, const State&, const std::vector<Species>&, const FMTModel&) -> Contribution` |
| `mean_field` | `auto mean_field(const Grid&, const State&, const std::vector<Interaction>&, double kT) -> Contribution` |
| `external_field` | `auto external_field(const Grid&, const State&) -> Contribution` |
| `total` | `auto total(const Model&, const State&) -> Result` |
| `constrain_mass` | `auto constrain_mass(const Grid&, SpeciesState, double target) -> SpeciesState` |

#### `dft::functionals::bulk`

| Function | Signature |
|---|---|
| `pressure` | `auto pressure(const Model&, double rho, double kT) -> double` |
| `chemical_potential` | `auto chemical_potential(const Model&, double rho, double kT) -> double` |
| `grand_potential` | `auto grand_potential(const Model&, double rho, double kT) -> double` |
| `helmholtz_free_energy` | `auto helmholtz_free_energy(const Model&, double rho, double kT) -> double` |
| `coexistence` | `auto coexistence(const Model&, double T, double rho_v_guess, double rho_l_guess) -> std::optional<PhasePoint>` |
| `binodal` | `auto binodal(const Model&, double start_T, double start_rho_v, double start_rho_l, const ContinuationConfig&) -> std::vector<PhasePoint>` |
| `spinodal` | `auto spinodal(const Model&, double start_T, double start_rho, const ContinuationConfig&) -> std::vector<PhasePoint>` |

#### `dft::algorithms`

| Function | Signature |
|---|---|
| `fire` | `auto fire(const Model&, State, const FireConfig&, StepCallback) -> Solution` |
| `split_operator` | `auto split_operator(const Model&, State, const SplitOperatorConfig&, StepCallback) -> Solution` |
| `crank_nicholson` | `auto crank_nicholson(const Model&, State, const CrankNicholsonConfig&, StepCallback) -> Solution` |
| `to_density` | `auto to_density(const AliasTransform&, const arma::vec& x) -> arma::vec` |
| `to_alias` | `auto to_alias(const AliasTransform&, const arma::vec& rho) -> arma::vec` |
| `chain_rule` | `auto chain_rule(const AliasTransform&, const arma::vec& x, const arma::vec& dF_drho) -> arma::vec` |

#### `dft::algorithms::solvers`

| Function | Signature |
|---|---|
| `newton` | `auto newton(arma::vec x, Func&& f, JacFunc&& J, const NewtonConfig&) -> SolverResult` |
| `newton` (auto-jacobian) | `auto newton(arma::vec x, Func&& f, const NewtonConfig&) -> SolverResult` |
| `numerical_jacobian` | `auto numerical_jacobian(Func&& f, const arma::vec& x, double eps) -> arma::mat` |

#### `dft::algorithms::continuation`

| Function | Signature |
|---|---|
| `step` | `auto step(const CurvePoint&, const Residual&, double ds, const ContinuationConfig&) -> std::optional<CurvePoint>` |
| `trace` | `auto trace(CurvePoint start, const Residual&, const ContinuationConfig&, StopCondition) -> std::vector<CurvePoint>` |

#### `dft::math`

| Function | Signature |
|---|---|
| `convolve` | `auto convolve(const FourierTransform& weight, std::span<const std::complex<double>> rho_k, FourierTransform& scratch) -> arma::vec` |
| `accumulate` | `void accumulate(const FourierTransform& weight, const arma::vec& deriv, FourierTransform& scratch, std::span<std::complex<double>> force_k)` |

#### `dft::geometry`

| Function | Signature |
|---|---|
| `uniform_mesh_2d` | `auto uniform_mesh_2d(double dx, ...) -> UniformMesh2D` |
| `uniform_mesh_3d` | `auto uniform_mesh_3d(double dx, ...) -> UniformMesh3D` |
| `volume` | `auto volume(const Mesh&) -> double` |
| `element_volume` | `auto element_volume(const Mesh&) -> double` |
| `flat_index` | `auto flat_index(const Mesh&, const std::vector<long>&) -> long` |
| `cartesian_index` | `auto cartesian_index(const Mesh&, long) -> std::vector<long>` |
| `vertex` | `auto vertex(const Mesh&, const std::vector<long>&) -> const Vertex&` |
| `plot` | `void plot(const Mesh&, const std::string& path, bool interactive)` |
| `wrap` | `auto wrap(const Mesh&, const Vertex&) -> Vertex` |
| `spacing` | `auto spacing(const Mesh&) -> double` |

---

## 5. Phase 1: core data structures (`core/`)

### 5.1. `core/grid.hpp`

Lightweight `Grid` struct for the DFT pipeline. Holds the three values that
`Density` currently stores privately (`dx_`, `box_size_`, `shape_`). This does
not replace the geometry module; it is a focused struct for DFT algorithms.

```cpp
namespace dft::core {
    struct Grid {
        double dx;
        std::array<double, 3> box_size;
        std::array<long, 3> shape;

        [[nodiscard]] constexpr auto cell_volume() const noexcept -> double;
        [[nodiscard]] constexpr auto total_points() const noexcept -> long;
        [[nodiscard]] constexpr auto flat_index(long ix, long iy, long iz) const noexcept -> long;
    };

    [[nodiscard]] auto make_grid(double dx, std::array<double, 3> box) -> Grid;
}
```

`make_grid` validates commensurateness (box size must be integer multiple of
dx) and computes `shape`.

### 5.2. `core/density.hpp`

Replaces `include/dft/density.h` + `src/density.cpp`. Strips all behavior.

```cpp
namespace dft::core {
    struct Density {
        arma::vec values;
        arma::vec external_field;
    };
}
```

Current methods become free functions:

| Old method | New free function | Location |
|---|---|---|
| `Density::forward_fft()` | `math::forward_fft(grid, density.values)` | `math/fourier.hpp` |
| `Density::number_of_atoms()` | `core::atom_count(grid, density)` | `core/grid.hpp` |
| `Density::center_of_mass()` | `core::center_of_mass(grid, density)` | `core/grid.hpp` |
| `Density::external_field_energy()` | `functionals::external_field(grid, state)` | `functionals/external_field.hpp` |
| `Density::save()` / `load()` | `io::save_density(path, field)` / `io::load_density(path)` | TBD |
| `Density::min()` / `max()` | `arma::min(density.values)` / `arma::max(density.values)` | direct Armadillo |

### 5.3. `core/species.hpp`

Replaces `include/dft/species.h` + `src/species.cpp`. The old `Species` class
mixed immutable physical identity with mutable simulation state. Split into two
structs:

```cpp
namespace dft::core {
    struct Species {
        std::string name;
        double hard_sphere_diameter;
    };

    struct SpeciesState {
        Density density;
        arma::vec force;
        double chemical_potential{0.0};
        std::optional<double> fixed_mass;
    };
}
```

Logic from old `Species` moves to:

| Old method | New location |
|---|---|
| `set_density_from_alias()` / `density_alias()` / `alias_force()` | `algorithms/alias.hpp` |
| `begin_force_calculation()` / `end_force_calculation()` | `functionals::constrain_mass()` |
| `convergence_monitor()` | `double max_force(const SpeciesState&, const Grid&)` free function |
| `external_field_energy()` | `functionals::external_field()` |

### 5.4. `core/state.hpp`

```cpp
namespace dft::core {
    struct State {
        std::vector<SpeciesState> species;
        double temperature;
    };
}
```

### 5.5. `init.hpp`

Convenience state factories:

```cpp
namespace dft::init {
    [[nodiscard]] auto homogeneous(const physics::Model&, double density) -> core::State;
    [[nodiscard]] auto from_profile(const physics::Model&, const arma::vec& rho) -> core::State;
    [[nodiscard]] auto from_file(const physics::Model&, const std::string& path) -> core::State;
}
```

---

## 6. Phase 1b: geometry modernization (`geometry/`)

The 3+ level virtual hierarchy (`Element` -> `SquareBox` -> `2D::SquareBox`,
`Mesh` -> `SUQMesh` -> `2D::SUQMesh` -> `2D::UniformMesh`) is flattened into
value types with `std::variant`. All functionality is preserved.

### 6.1. `geometry/vertex.hpp`

`Vertex` is already effectively a value type (no virtuals). Make
`coordinates` public. Remove the redundant `dimension_` member (infer from
`coordinates.size()`). Keep `operator+`, `operator-`, `operator[]`,
`operator<<`.

### 6.2. `geometry/element.hpp`

Replace the `Element` -> `SquareBox` hierarchy with flat structs:

```cpp
namespace dft::geometry {
    struct Element {
        std::vector<Vertex> vertices;
    };

    struct SquareBox2D {
        double length;
        std::vector<double> origin;
        std::vector<Vertex> vertices;
    };

    struct SquareBox3D {
        double length;
        std::vector<double> origin;
        std::vector<Vertex> vertices;
    };

    using ElementVariant = std::variant<Element, SquareBox2D, SquareBox3D>;

    [[nodiscard]] auto volume(const ElementVariant&) -> double;
    [[nodiscard]] auto dimension(const ElementVariant&) -> int;
}
```

### 6.3. `geometry/mesh.hpp`

Replace the `Mesh` -> `SUQMesh` -> `UniformMesh` hierarchy:

```cpp
namespace dft::geometry {
    struct UniformMesh2D {
        double dx;
        std::vector<double> dimensions;
        std::vector<double> origin;
        std::vector<long> shape;
        std::vector<Vertex> vertices;
        std::vector<SquareBox2D> elements;
    };

    struct UniformMesh3D {
        double dx;
        std::vector<double> dimensions;
        std::vector<double> origin;
        std::vector<long> shape;
        std::vector<Vertex> vertices;
        std::vector<SquareBox3D> elements;
    };

    using Mesh = std::variant<UniformMesh2D, UniformMesh3D>;
}
```

All current public operations preserved as free functions: `volume()`,
`element_volume()`, `flat_index()`, `cartesian_index()`, `vertex()`, `plot()`,
`wrap()`, `spacing()`, `uniform_mesh_2d()`, `uniform_mesh_3d()`.

---

## 7. Phase 2: math utilities (`math/`)

### 7.1. Files kept with minor rename (.h -> .hpp)

- `fourier.hpp` — `FourierTransform` and `FourierConvolution` (RAII, already
  well designed).
- `arithmetic.hpp` — compensated summation (already functional style).
- `spline.hpp` — `CubicSpline` and `BivariateSpline` (RAII GSL wrappers).
- `integration.hpp` — `Integrator<T>` (template-based, already clean).
- `autodiff.hpp` — type aliases and derivative helpers (kept as-is).

### 7.2. `math/hessian.hpp` (kept and modernized)

The `HessianOperator` abstract class becomes either a C++20 concept or a
type-erased callable:

```cpp
namespace dft::math {
    struct HessianOperator {
        std::function<void(const arma::vec&, arma::vec&)> hessian_dot_v;
        arma::uword dimension;
    };
}
```

### 7.3. `math/convolution.hpp` (new, extracted from ConvolutionField)

Pure functions for FFT-based convolution, extracted from
`functional/fmt/convolution.h`:

```cpp
namespace dft::math {
    [[nodiscard]] auto convolve(
        const FourierTransform& weight,
        std::span<const std::complex<double>> rho_k,
        FourierTransform& scratch
    ) -> arma::vec;

    void accumulate(
        const FourierTransform& weight,
        const arma::vec& derivative,
        FourierTransform& scratch,
        std::span<std::complex<double>> force_k
    );
}
```

### 7.4. `exceptions.hpp` (kept)

`WrongParameterException` and `NegativeParameterException` preserved. Grace
exceptions preserved in `plotting/exceptions.hpp`.

---

## 8. Phase 2b: generic solvers (`algorithms/solvers/`)

Modernizes the old Numerix library code. The mathematical core (Newton-Raphson
iteration loop, central-difference Jacobian, pseudo-arclength augmented system)
is preserved. The class hierarchy, `shared_ptr` ownership, string-based
factory, getters/setters, and `GENERATE_HAS_MEMBER` macro are eliminated.

### 8.1. `algorithms/solvers/jacobian.hpp`

Modernizes `Numerix::NumericalJacobian` from `Functions.h`. Central-difference
O(h^2) scheme preserved. Member-function-pointer signature replaced by C++20
concept:

```cpp
namespace dft::algorithms::solvers {
    template <typename F>
    concept VectorFunction = requires(F f, const arma::vec& x) {
        { f(x) } -> std::convertible_to<arma::vec>;
    };

    template <VectorFunction Func>
    [[nodiscard]] auto numerical_jacobian(
        Func&& f, const arma::vec& x, double epsilon = 1e-7
    ) -> arma::mat;
}
```

### 8.2. `algorithms/solvers/newton.hpp`

Modernizes `Numerix::NewtonSolver::Solve()`. The loop
(`arma::solve(Jk, fk)` -> `x -= delta` -> check norm) is preserved exactly:

```cpp
namespace dft::algorithms::solvers {
    template <typename J>
    concept JacobianFunction = requires(J jac, const arma::vec& x) {
        { jac(x) } -> std::convertible_to<arma::mat>;
    };

    struct NewtonConfig {
        int max_iterations{100};
        double tolerance{1e-6};
        bool verbose{false};
    };

    struct SolverResult {
        arma::vec solution;
        int iterations;
        double final_norm;
        bool converged;
    };

    // With analytical Jacobian
    template <VectorFunction Func, JacobianFunction JacFunc>
    [[nodiscard]] auto newton(
        arma::vec x, Func&& f, JacFunc&& J, const NewtonConfig& config = {}
    ) -> SolverResult;

    // Without analytical Jacobian (generates numerical Jacobian internally)
    template <VectorFunction Func>
    [[nodiscard]] auto newton(
        arma::vec x, Func&& f, const NewtonConfig& config = {}
    ) -> SolverResult;
}
```

Mapping from Numerix:

| Numerix | New |
|---|---|
| `SolverKernel::m_MaxIter` | `NewtonConfig::max_iterations` |
| `SolverKernel::m_EpsRel` | `NewtonConfig::tolerance` |
| `SolverKernel::m_Verbose` | `NewtonConfig::verbose` |
| `NewtonSolver::Solve()` loop | body of `newton()` |
| `HasJacobian<T>::value` macro | second overload without `JacFunc` |
| Return initial guess on failure | `SolverResult{.converged = false}` |

### 8.3. `algorithms/solvers/continuation.hpp`

Completes the unfinished `Numerix::MoorePenrose` as pseudo-arclength
continuation. The old code had tangent `m_tau`, step size `m_ds`, and augmented
system `N+1`, but `Solve()` was never implemented.

```cpp
namespace dft::algorithms::continuation {
    struct CurvePoint {
        arma::vec x;
        double lambda;
        arma::vec dx_ds;
        double dlambda_ds;
    };

    struct ContinuationConfig {
        double initial_step{0.01};
        double max_step{0.1};
        double min_step{1e-5};
        double growth_factor{1.2};
        double shrink_factor{0.5};
        solvers::NewtonConfig newton;
    };

    using Residual = std::function<arma::vec(const arma::vec&, double)>;

    [[nodiscard]] auto step(
        const CurvePoint& current, const Residual& R, double ds,
        const ContinuationConfig& config
    ) -> std::optional<CurvePoint>;

    [[nodiscard]] auto trace(
        CurvePoint start, const Residual& R,
        const ContinuationConfig& config,
        std::function<bool(const CurvePoint&)> stop = {}
    ) -> std::vector<CurvePoint>;
}
```

How `step()` works:

1. Predictor: Euler along tangent. `x_pred = current.x + ds * current.dx_ds`.
2. Pack into augmented vector `y = [x_pred; lambda_pred]`.
3. Build augmented residual: physics R(x, lambda) + arclength constraint.
4. Build augmented Jacobian: extended [dR/dx | dR/dlambda] + constraint row.
5. Correct via `solvers::newton(y_pred, augmented_residual, augmented_jacobian)`.
6. Compute new tangent from null space of extended Jacobian. Orient to match
   previous direction.
7. Return `CurvePoint` or `std::nullopt` if Newton failed.

`trace()` is the outer loop with adaptive step sizing: grow on success, shrink
on failure, stop when condition met or step too small.

---

## 9. Phase 3: physics definitions (`physics/`)

### 9.1. `physics/potentials.hpp`

Replaces the `Potential` virtual hierarchy (11 protected members, 4 pure
virtual methods, 3 concrete subclasses):

```cpp
namespace dft::physics::potentials {
    struct LennardJones { double sigma; double epsilon; double r_cutoff; };
    struct TenWoldeFrenkel { double sigma; double epsilon; double r_cutoff; double alpha; };
    struct WangRamirezDobnikarFrenkel { double sigma; double epsilon; double r_cutoff; };

    using Potential = std::variant<LennardJones, TenWoldeFrenkel, WangRamirezDobnikarFrenkel>;

    struct SplitScheme {
        bool barker_henderson{false};
        double r_attractive_min{0.0};
    };
}
```

All methods become free functions dispatched via `std::visit`: `energy()`,
`energy_r2()`, `attractive()`, `repulsive()`, `hard_sphere_diameter()`,
`vdw_integral()`, `r_min()`, `hard_core_diameter()`.

### 9.2. `physics/interactions.hpp`

Replaces `functional/interaction.h`. The `Interaction` class (which held
species references, FFT weights, and computed its own forces) becomes a pure
specification struct:

```cpp
namespace dft::physics {
    struct Interaction {
        std::size_t species_a;
        std::size_t species_b;
        potentials::Potential potential;
        potentials::SplitScheme split_scheme;
        WeightScheme weight_scheme{WeightScheme::InterpolationLinearF};
        int gauss_order{5};
    };
}
```

The actual computation moves to `functionals/mean_field.hpp`.

### 9.3. `physics/fmt/models.hpp`

Keeps the existing `Rosenfeld`, `RSLT`, `WhiteBearI`, `WhiteBearII` structs
(already stateless with static template methods). Keeps
`FMTModel = std::variant<...>`.

Eliminates the `FMT` wrapper class. Its methods become free functions:
`free_energy_density()`, `derivatives()`, `bulk_free_energy()`,
`excess_chemical_potential()`, `requires_tensor()`, `name()`.

### 9.4. `physics/fmt/measures.hpp`

Kept as-is. `Measures` and `InnerProducts` are already pure public structs.

### 9.5. `physics/fmt/weights.hpp`

`WeightSet` changes to hold `FourierTransform` objects directly instead of
`ConvolutionField` objects:

```cpp
struct FmtWeightSet {
    FourierTransform eta;
    FourierTransform scalar;
    std::array<FourierTransform, 3> vector;
    std::array<std::array<FourierTransform, 3>, 3> tensor;
};

[[nodiscard]] auto weights(double diameter, const Grid& grid) -> FmtWeightSet;
```

Analytic Fourier weight formulas (`volume_hat`, `surface_hat`,
`vector_prefactor`, `tensor_coefficients`) preserved from current
`Weights::generate()`.

### 9.6. `physics/model.hpp`

Replaces the `Solver` god-object:

```cpp
namespace dft::physics {
    struct Model {
        core::Grid grid;
        std::vector<core::Species> species;
        std::vector<Interaction> interactions;
        std::optional<fmt::FMTModel> fmt;
    };
}
```

Pure data aggregate. Does not hold density arrays, force arrays, or FFT
buffers.

---

## 10. Phase 4: functionals (`functionals/`)

### 10.1. `functionals/ideal_gas.hpp`

Extracted from `Solver::compute_free_energy_and_forces()`:

```cpp
namespace dft::functionals {
    struct Contribution {
        double free_energy;
        std::vector<arma::vec> forces;
    };

    [[nodiscard]] auto ideal_gas(const core::Grid&, const core::State&) -> Contribution;
}
```

### 10.2. `functionals/hard_sphere.hpp`

Extracts the forward-backward FMT pipeline from `fmt::Species::compute_forces()`:

```cpp
[[nodiscard]] auto hard_sphere(
    const core::Grid&, const core::State&,
    const std::vector<core::Species>&, const physics::fmt::FMTModel&
) -> Contribution;
```

Pre-computed workspace for iterative algorithms:

```cpp
struct FmtWorkspace {
    std::vector<physics::fmt::FmtWeightSet> weights;
    std::vector<math::fourier::FourierTransform> scratch;
};

[[nodiscard]] auto make_fmt_workspace(
    const core::Grid&, const std::vector<core::Species>&
) -> FmtWorkspace;
```

### 10.3. `functionals/mean_field.hpp`

Extracted from `Interaction::compute_forces()`:

```cpp
[[nodiscard]] auto mean_field(
    const core::Grid&, const core::State&,
    const std::vector<physics::Interaction>&, double kT
) -> Contribution;
```

### 10.4. `functionals/external_field.hpp`

Extracted from `Species::external_field_energy()`:

```cpp
[[nodiscard]] auto external_field(const core::Grid&, const core::State&) -> Contribution;
```

### 10.5. `functionals/functionals.hpp`

Orchestrator replacing `Solver::compute_free_energy_and_forces()`:

```cpp
namespace dft::functionals {
    struct Result {
        double total_free_energy;
        double ideal;
        double hard_sphere;
        double mean_field;
        double external;
        std::vector<arma::vec> forces;
    };

    [[nodiscard]] auto total(const physics::Model&, const core::State&) -> Result;
}
```

### 10.6. `functionals/bulk/thermodynamics.hpp`

```cpp
namespace dft::functionals::bulk {
    [[nodiscard]] auto pressure(const physics::Model&, double rho, double kT) -> double;
    [[nodiscard]] auto chemical_potential(const physics::Model&, double rho, double kT) -> double;
    [[nodiscard]] auto grand_potential(const physics::Model&, double rho, double kT) -> double;
    [[nodiscard]] auto helmholtz_free_energy(const physics::Model&, double rho, double kT) -> double;
}
```

### 10.7. `functionals/bulk/phase_diagram.hpp`

Replaces the current 170 lines of golden-section search + bisection in
`solver.cpp` (lines 224-390) with continuation-based tracing:

```cpp
namespace dft::functionals::bulk {
    struct PhasePoint {
        double rho_vapor;
        double rho_liquid;
        double temperature;
    };

    [[nodiscard]] auto binodal(
        const physics::Model&, double start_T, double start_rho_v,
        double start_rho_l, const algorithms::continuation::ContinuationConfig&
    ) -> std::vector<PhasePoint>;

    [[nodiscard]] auto spinodal(
        const physics::Model&, double start_T, double start_rho,
        const algorithms::continuation::ContinuationConfig&
    ) -> std::vector<PhasePoint>;

    [[nodiscard]] auto coexistence(
        const physics::Model&, double T, double rho_v_guess, double rho_l_guess
    ) -> std::optional<PhasePoint>;
}
```

`binodal()` internally builds a `Residual` lambda capturing the `Model`:

```
R(state, T) = [P(rho_v, T) - P(rho_l, T),  mu(rho_v, T) - mu(rho_l, T)]
```

Then calls `continuation::trace(start, R, config, stop)` where `stop` checks
`|rho_v - rho_l| < threshold` (critical point reached).

`coexistence()` uses `solvers::newton` directly for a single temperature.

---

## 11. Phase 5: algorithms (`algorithms/`)

### 11.1. `algorithms/alias.hpp`

Replaces all virtual alias methods with variant-based transforms:

```cpp
namespace dft::algorithms {
    struct UnboundedAlias { double rho_min{1e-18}; };
    struct BoundedAlias { double rho_min{1e-18}; double rho_max; };

    using AliasTransform = std::variant<UnboundedAlias, BoundedAlias>;

    [[nodiscard]] auto to_density(const AliasTransform&, const arma::vec& x) -> arma::vec;
    [[nodiscard]] auto to_alias(const AliasTransform&, const arma::vec& rho) -> arma::vec;
    [[nodiscard]] auto chain_rule(const AliasTransform&, const arma::vec& x, const arma::vec& dF_drho) -> arma::vec;
}
```

- `UnboundedAlias`: rho = rho_min + x^2 (current `Species` base).
- `BoundedAlias`: rho = rho_min + range * y^2 / (1 + y^2) (current
  `fmt::Species` override).

This eliminates the only reason `fmt::Species` inherits from `Species`.

### 11.2. `algorithms/fire.hpp`

Replaces `dynamics/fire2.h` + `src/dynamics/fire2.cpp`:

```cpp
namespace dft::algorithms {
    using StepCallback = std::function<bool(long step, double energy, double max_force)>;

    struct FireConfig {
        double dt{1e-3};
        double dt_max{1e-2};
        double dt_min{1e-8};
        double alpha_start{0.1};
        double f_inc{1.1};
        double f_dec{0.5};
        double f_alf{0.99};
        int n_delay{5};
        int max_uphill_steps{20};
        double force_tolerance{0.1};
        double min_density{1e-30};
        long max_steps{0};
    };

    struct Solution {
        core::State state;
        functionals::Result evaluation;
        long steps;
        bool converged;
    };

    [[nodiscard]] auto fire(
        const physics::Model&, core::State, const FireConfig&, StepCallback = {}
    ) -> Solution;
}
```

Takes `State` by value, runs the FIRE2 loop, returns final state. Velocity
arrays are local variables inside the function.

### 11.3. `algorithms/split_operator.hpp`

```cpp
struct SplitOperatorConfig {
    double dt{1e-4};
    double diffusion_coefficient{1.0};
    double force_tolerance{0.1};
    double min_density{1e-30};
    long max_steps{0};
};

[[nodiscard]] auto split_operator(
    const physics::Model&, core::State, const SplitOperatorConfig&, StepCallback = {}
) -> Solution;
```

### 11.4. `algorithms/crank_nicholson.hpp`

```cpp
struct CrankNicholsonConfig {
    double dt{1e-4};
    double diffusion_coefficient{1.0};
    int inner_iterations{5};
    double cn_tolerance{1e-10};
    double force_tolerance{0.1};
    double min_density{1e-30};
    long max_steps{0};
};

[[nodiscard]] auto crank_nicholson(
    const physics::Model&, core::State, const CrankNicholsonConfig&, StepCallback = {}
) -> Solution;
```

### 11.5. Minimizer base class eliminated

`include/dft/dynamics/minimizer.h` and `src/dynamics/minimizer.cpp` are
eliminated. The NVI pattern (public `run()` -> protected virtual `do_step()`)
is replaced by standalone free functions.

---

## 12. Phase 6: peripheral modules

### 12.1. `crystal/`

Rename `.h` -> `.hpp`. Minor cleanup: `Lattice::build()` becomes a free
function returning positions + dimensions instead of a constructor side-effect.

### 12.2. `config/`

Rename `config.h` -> `config/parser.hpp`. Remove setters. Make `ConfigParser`
constructible only with file path + type. Or simpler: free function
`auto parse_config(path, type) -> nlohmann::json`.

### 12.3. `console.hpp`

Keep as-is. Stateless utility namespace.

### 12.4. `plotting/`

Keep but decouple from core. `Grace` class manages an external process (pipe
to xmgrace) and is inherently stateful. Leave it as a class. No core code
depends on it.

### 12.5. Exception classes

`plotting/exceptions.hpp` kept. `exceptions.hpp` kept.

---

## 13. Phase 7: testing migration (GTest -> Catch2 v3)

### 13.1. CMake changes

Replace `FetchContent` for `googletest v1.14.0` with:

```cmake
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)
```

Link `Catch2::Catch2WithMain` instead of `GTest::gtest`. Delete
`tests/main.cpp` (Catch2 provides its own main).

### 13.2. Macro translation

| GTest | Catch2 |
|---|---|
| `TEST(Suite, Name)` | `TEST_CASE("Name", "[tag]")` |
| `TEST_F(Fixture, Name)` | `TEST_CASE("Name") { setup; SECTION("...") {} }` |
| `TEST_P` + `INSTANTIATE_TEST_SUITE_P` | `GENERATE(table<...>({...}))` |
| `EXPECT_EQ(a, b)` | `CHECK(a == b)` |
| `ASSERT_EQ(a, b)` | `REQUIRE(a == b)` |
| `EXPECT_NEAR(a, b, tol)` | `CHECK(a == Catch::Approx(b).margin(tol))` |
| `EXPECT_DOUBLE_EQ(a, b)` | `CHECK(a == Catch::Approx(b))` |
| `EXPECT_THROW(expr, type)` | `REQUIRE_THROWS_AS(expr, type)` |
| `EXPECT_TRUE(x)` | `CHECK(x)` |
| `EXPECT_LE(a, b)` | `CHECK(a <= b)` |

### 13.3. New test directory structure

```
tests/
  core/
    density.cpp
    species.cpp
    state.cpp
  math/
    arithmetic.cpp
    fourier.cpp
    convolution.cpp
    integration.cpp
    spline.cpp
  physics/
    potentials.cpp
    enskog.cpp
    eos.cpp
    fmt/
      models.cpp
      measures.cpp
      weights.cpp
  functionals/
    functionals.cpp
    hard_sphere.cpp
    mean_field.cpp
    ideal_gas.cpp
    bulk/
      phase_diagram.cpp
  algorithms/
    fire.cpp
    split_operator.cpp
    crank_nicholson.cpp
    alias.cpp
    solvers/
      newton.cpp
      jacobian.cpp
      continuation.cpp
  geometry/
    vertex.cpp
    element.cpp
    mesh.cpp
  crystal/
    lattice.cpp
  config/
    parser.cpp
  console.cpp
  plotting/
    grace.cpp
    exceptions.cpp
```

32 current files -> 28 files (geometry consolidation 7->3, delete main.cpp,
add new test files for solvers/continuation/phase_diagram).

### 13.4. Test migration by file

**Core:**

| Current | New | Notes |
|---|---|---|
| `tests/density.cpp` | `tests/core/density.cpp` | Test free functions |
| `tests/species.cpp` | `tests/core/species.cpp` | Alias tests -> `tests/algorithms/alias.cpp` |
| `tests/solver.cpp` | `tests/functionals/functionals.cpp` | Pipeline tests |
| `tests/config.cpp` | `tests/config/parser.cpp` | `TEST_P` -> `GENERATE` |

**Math (mostly 1:1):**

| Current | New |
|---|---|
| `tests/math/arithmetic.cpp` | `tests/math/arithmetic.cpp` |
| `tests/math/fourier.cpp` | `tests/math/fourier.cpp` |
| `tests/math/integration.cpp` | `tests/math/integration.cpp` |
| `tests/math/spline.cpp` | `tests/math/spline.cpp` |

**Geometry (7 files -> 3):**

| Current | New |
|---|---|
| `tests/geometry/base/vertex.cpp` | `tests/geometry/vertex.cpp` |
| `tests/geometry/base/element.cpp` + `2D/element.cpp` + `3D/element.cpp` | `tests/geometry/element.cpp` |
| `tests/geometry/2D/mesh.cpp` + `2D/uniform_mesh.cpp` + `3D/mesh.cpp` + `3D/uniform_mesh.cpp` | `tests/geometry/mesh.cpp` |

**Physics:**

| Current | New |
|---|---|
| `tests/potentials/potential.cpp` | `tests/physics/potentials.cpp` |
| `tests/thermodynamics/enskog.cpp` | `tests/physics/enskog.cpp` |
| `tests/thermodynamics/eos.cpp` | `tests/physics/eos.cpp` |

**FMT:**

| Current | New |
|---|---|
| `tests/functional/fmt/functional.cpp` | `tests/physics/fmt/models.cpp` |
| `tests/functional/fmt/measures.cpp` | `tests/physics/fmt/measures.cpp` |
| `tests/functional/fmt/convolution.cpp` | `tests/math/convolution.cpp` |
| `tests/functional/fmt/weights.cpp` | `tests/physics/fmt/weights.cpp` |
| `tests/functional/fmt/species.cpp` | `tests/functionals/hard_sphere.cpp` |
| `tests/functional/interaction.cpp` | `tests/functionals/mean_field.cpp` |

**Dynamics (1 file -> 3):**

| Current | New |
|---|---|
| `tests/dynamics/minimizer.cpp` | `tests/algorithms/fire.cpp` |
| | `tests/algorithms/split_operator.cpp` |
| | `tests/algorithms/crank_nicholson.cpp` |

**New test files:**

| File | Tests |
|---|---|
| `tests/algorithms/solvers/newton.cpp` | Newton on known systems; with/without Jacobian |
| `tests/algorithms/solvers/jacobian.cpp` | Numerical vs analytical Jacobian accuracy |
| `tests/algorithms/solvers/continuation.cpp` | Trace known curve around turning point |
| `tests/functionals/bulk/phase_diagram.cpp` | Binodal/spinodal for CS + LJ at known T |
| `tests/algorithms/alias.cpp` | Alias roundtrip invertibility |
| `tests/functionals/ideal_gas.cpp` | Ideal gas energy and forces |

### 13.5. Test migration order (bottom-up)

1. `tests/math/` — pure functions, no deps on new architecture
2. `tests/geometry/` — depends on geometry modernization
3. `tests/core/` — depends on Phase 1
4. `tests/physics/` — depends on Phase 3
5. `tests/functionals/` — depends on Phase 4
6. `tests/algorithms/` — depends on Phase 5
7. `tests/crystal/`, `tests/config/`, `tests/console/`, `tests/plotting/` —
   anytime

---

## 14. File inventory

### 14.1. Files to create (28 new headers + 1 init)

| File | Content |
|---|---|
| `core/grid.hpp` | `Grid` struct + helpers |
| `core/density.hpp` | `Density` data struct |
| `core/species.hpp` | `Species` + `SpeciesState` |
| `core/state.hpp` | `State` aggregate |
| `init.hpp` | Convenience state factories |
| `math/convolution.hpp` | `convolve()`, `accumulate()` |
| `physics/potentials.hpp` | Variant-based potentials |
| `physics/interactions.hpp` | `Interaction` spec struct |
| `physics/fmt/models.hpp` | FMT model structs + free fns |
| `physics/fmt/measures.hpp` | Renamed from current |
| `physics/fmt/weights.hpp` | `FmtWeightSet` + `weights()` |
| `physics/model.hpp` | `Model` aggregate |
| `functionals/ideal_gas.hpp` | `ideal_gas()` |
| `functionals/hard_sphere.hpp` | `hard_sphere()` + `FmtWorkspace` |
| `functionals/mean_field.hpp` | `mean_field()` |
| `functionals/external_field.hpp` | `external_field()` |
| `functionals/functionals.hpp` | `total()` + `Result` |
| `functionals/bulk/thermodynamics.hpp` | Bulk pure functions |
| `functionals/bulk/phase_diagram.hpp` | Continuation-based phase tracing |
| `algorithms/solvers/newton.hpp` | Generic Newton-Raphson |
| `algorithms/solvers/jacobian.hpp` | Numerical Jacobian |
| `algorithms/solvers/continuation.hpp` | Pseudo-arclength continuation |
| `algorithms/alias.hpp` | `AliasTransform` variant |
| `algorithms/fire.hpp` | `FireConfig` + `fire()` |
| `algorithms/split_operator.hpp` | `SplitOperatorConfig` + fn |
| `algorithms/crank_nicholson.hpp` | `CrankNicholsonConfig` + fn |
| `geometry/vertex.hpp` | Value-type Vertex |
| `geometry/element.hpp` | Variant-based elements |
| `geometry/mesh.hpp` | Variant-based meshes |

### 14.2. Files kept (renamed .h -> .hpp)

| Current | New |
|---|---|
| `math/fourier.h` | `math/fourier.hpp` |
| `math/arithmetic.h` | `math/arithmetic.hpp` |
| `math/spline.h` | `math/spline.hpp` |
| `math/integration.h` | `math/integration.hpp` |
| `math/autodiff.h` | `math/autodiff.hpp` |
| `math/hessian.h` | `math/hessian.hpp` |
| `math/exceptions.h` | `exceptions.hpp` |
| `console.h` | `console.hpp` |
| `config.h` | `config/parser.hpp` |
| `crystal/lattice.h` | `crystal/lattice.hpp` |
| `crystal/types.h` | `crystal/types.hpp` |
| `plotting/grace.h` | `plotting/grace.hpp` |
| `plotting/exceptions.h` | `plotting/exceptions.hpp` |
| `thermodynamics/enskog.h` | `physics/enskog.hpp` |
| `thermodynamics/eos.h` | `physics/eos.hpp` |

Corresponding `.cpp` files kept for non-inline implementations: config parser,
fourier (FFTW allocation), spline (GSL allocation), arithmetic, crystal
lattice, grace (IPC).

### 14.3. Files eliminated (logic redistributed)

| File | Where logic goes |
|---|---|
| `density.h` / `density.cpp` | `core/density.hpp` (data) + free fns |
| `species.h` / `species.cpp` | `core/species.hpp` + `algorithms/alias.hpp` + `functionals/` |
| `solver.h` / `solver.cpp` | `physics/model.hpp` + `functionals/` + `functionals/bulk/` |
| `potentials/potential.h` / `.cpp` | `physics/potentials.hpp` |
| `potentials/types.h` | merged into `physics/potentials.hpp` |
| `functional/interaction.h` / `.cpp` | `physics/interactions.hpp` + `functionals/mean_field.hpp` |
| `functional/fmt/convolution.h` / `.cpp` | `math/convolution.hpp` |
| `functional/fmt/functional.h` | `physics/fmt/models.hpp` |
| `functional/fmt/species.h` / `.cpp` | `functionals/hard_sphere.hpp` + `algorithms/alias.hpp` |
| `dynamics/minimizer.h` / `.cpp` | algorithm free functions |
| `dynamics/integrator.h` / `.cpp` | `algorithms/split_operator.hpp` + `algorithms/crank_nicholson.hpp` |
| `dynamics/fire2.h` / `.cpp` | `algorithms/fire.hpp` |
| `geometry/base/element.h` / `.cpp` | `geometry/element.hpp` |
| `geometry/base/mesh.h` / `.cpp` | `geometry/mesh.hpp` |
| `geometry/2D/*.h` / `.cpp` | merged into `geometry/element.hpp` + `geometry/mesh.hpp` |
| `geometry/3D/*.h` / `.cpp` | merged into `geometry/element.hpp` + `geometry/mesh.hpp` |

---

## 15. Final API example

```cpp
#include <dft/dft.hpp>

int main() {
    using namespace dft;

    // 1. Define the system (declarative, all public data)
    physics::Model model{
        .grid = core::make_grid(0.1, {10.0, 10.0, 10.0}),
        .species = {
            core::Species{.name = "Argon", .hard_sphere_diameter = 1.0},
        },
        .interactions = {
            physics::Interaction{
                .species_a = 0,
                .species_b = 0,
                .potential = physics::potentials::LennardJones{
                    .sigma = 1.0, .epsilon = 1.0, .r_cutoff = 3.0
                },
            },
        },
        .fmt = physics::fmt::Rosenfeld{},
    };

    // 2. Initial state (one-liner via init:: factory)
    auto state = init::homogeneous(model, 0.5);

    // 3. Evaluate functionals (pure function, no side effects)
    auto result = functionals::total(model, state);
    std::cout << "Free energy: " << result.total_free_energy << "\n";

    // 4. Minimize (pure function, returns new state)
    auto solution = algorithms::fire(model, std::move(state), {
        .dt = 0.01,
        .force_tolerance = 1e-6,
        .max_steps = 10000,
    });

    // 5. Access results directly (no getters)
    std::cout << "Converged: " << solution.converged
              << " in " << solution.steps << " steps\n";
    std::cout << "Chemical potential: "
              << solution.state.species[0].chemical_potential << "\n";

    // 6. Bulk thermodynamics (pure functions)
    double mu = functionals::bulk::chemical_potential(model, 0.5, 1.0);
    auto [rho_gas, rho_liquid] = *functionals::bulk::coexistence(model, 1.0, 0.01, 0.8);

    // 7. Trace entire binodal via continuation
    auto envelope = functionals::bulk::binodal(model, 0.7, 0.01, 0.85, {
        .initial_step = 0.01,
        .max_step = 0.05,
        .newton = {.tolerance = 1e-10},
    });

    for (auto& p : envelope) {
        std::cout << p.temperature << " " << p.rho_vapor << " " << p.rho_liquid << "\n";
    }

    // 8. Use generic Newton solver for any custom problem
    auto root = algorithms::solvers::newton(
        arma::vec{1.0, 1.0},
        [](const arma::vec& x) { return arma::square(x) - 4.0; },
        [](const arma::vec& x) { return arma::diagmat(2.0 * x); },
        {.tolerance = 1e-12}
    );
}
```

---

## 16. Execution order

```
Phase 1   (core data)         ── no deps
Phase 1b  (geometry)          ── no deps, parallel with 1
Phase 2   (math)              ── depends on 1
Phase 2b  (solvers)           ── no deps (pure math, only armadillo)
Phase 3   (physics)           ── depends on 2
Phase 4   (functionals)       ── depends on 3 + 2b (bulk uses continuation)
Phase 5   (algorithms/dft)    ── depends on 4
Phase 6   (peripheral)        ── parallel anytime
Phase 7   (test migration)    ── after 5, parallel with 6
```

Within phases:

- Phase 1: steps 1.1-1.5 are independent.
- Phase 2: steps 2.1-2.4 are independent.
- Phase 2b: steps 2b.1-2b.3 are sequential (jacobian -> newton -> continuation).
- Phase 3: 3.1 and 3.3/3.4/3.5 independent; 3.2 depends on 3.1; 3.6 depends
  on all.
- Phase 4: 4.1-4.4 independent; 4.5 depends on all; 4.6-4.7 depend on 4.5
  and 2b.
- Phase 5: 5.1 independent; 5.2-5.4 depend on 4.5.
- Phase 7: bottom-up following dependency order (math -> geometry -> core ->
  physics -> functionals -> algorithms -> peripheral).

---

## 17. Verification criteria

1. **Numerical consistency**: after each phase, run the test suite. Free energy
   values, density profiles, and force magnitudes must match the current
   baseline to machine precision.
2. **Compilation**: all `std::visit` calls must handle every variant alternative
   (compiler enforces exhaustiveness).
3. **No virtual**: `grep -r "virtual" include/dft/` returns zero hits after
   completion (excluding geometry if in-progress).
4. **No getters/setters**: `grep -rE "void set_|get_" include/dft/` returns zero
   hits (excluding `plotting/grace.hpp`).
5. **No compute_ prefix**: `grep -r "compute_" include/dft/` returns zero hits.
6. **Memory**: Clang-Tidy `performance-*` checks pass. Zero unintentional copies
   of `arma::vec`. Verify with `-fsanitize=address`.
7. **Test coverage**: all ~250 existing tests pass under Catch2 with identical
   tolerances.
