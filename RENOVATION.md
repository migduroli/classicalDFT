# classicalDFT renovation plan (v2 — minimalistic OOP)

This document is the living architectural plan for the classicalDFT library.
It replaces the v1 plan that led to a procedural-in-disguise design. The new
plan embraces minimalistic OOP: types own their behavior, templates eliminate
boilerplate and hot-path overhead, and `class` with private members is used
when encapsulation genuinely serves the user.

---

## 1. Motivation

The v1 renovation moved away from deep inheritance hierarchies into
`struct + free functions + std::variant`. That was an improvement, but it
went too far: dogmatically banning `class`, `private`, and templates produced
~50 orphan free functions that always take a struct as their first argument —
procedural programming with extra steps. Users had to manually wire weights,
sync a_vdw values, and build force callbacks from scratch. The FMT model
choice floated free of the physics model.

**v2 principles:**
- **Types own their behavior.** `lj.energy(r)` not `energy(lj, r)`.
- **Templates for compile-time polymorphism** where it eliminates hot-path
  `std::visit` overhead or reduces boilerplate.
- **`class` with private members** when hiding implementation details (FFT
  weights, scratch buffers) genuinely simplifies the user API.
- **`struct` for pure data and configuration** — same as v1.
- **`std::variant` for runtime closed sets** — potentials in Model, mesh
  types — where the user doesn't choose at compile time.
- **No virtual, no inheritance** — same as v1.

---

## 2. Migration phases

### Phase 1: Potentials — methods on structs

Each potential struct (`LennardJones`, `TenWoldeFrenkel`,
`WangRamirezDobnikarFrenkel`) gains methods: `operator()`, `energy()`,
`repulsive()`, `attractive()`, `vdw_integral()`. Concrete-type free functions
become thin wrappers delegating to the methods. Variant dispatchers call
methods via `std::visit([r](const auto& p) { return p.energy(r); }, pot)`.

### Phase 2: Hard-sphere + EoS models — methods on structs

`CarnahanStarling`, `PercusYevickVirial`, `PercusYevickCompressibility` gain
`pressure()`, `free_energy()`, `chemical_potential()`. EoS structs
(`IdealGas`, `PercusYevick`, `LennardJonesJZG`, `LennardJonesMecke`) gain
the same. Variant dispatchers become thin wrappers.

### Phase 3: FMT models — methods on structs (HOT PATH)

`Rosenfeld`, `RSLT`, `WhiteBearI`, `WhiteBearII`, `EsFMT` gain `phi()`,
`d_phi()`, `free_energy_density()`, `excess_chemical_potential()` as
methods. This enables template-based hot-path optimization in Phase 6.

### Phase 4: Geometry + Grid — methods on structs

`Vertex::dimension()`, `UniformMesh2D/3D::volume()`, `element_volume()`,
`spacing()`. `SquareBox2D/3D::volume()`. `Grid::for_each_wavevector()`.
Variant dispatchers and concrete-type free functions become thin wrappers.

### Phase 5: Remaining orphan audit + bulk thermo

Final audit of all headers for orphan free functions. `PhaseDiagram`
gains `interpolate()` method. Confirmed that `BulkThermodynamics`,
`PhaseSearch`, `PhaseDiagramBuilder`, and `Functional` already own
their behavior from the prior v1 renovation. Factory functions
(`make_*`), multi-type orchestrators (`total()`), and operators stay free.

### Deferred: templatized `Functional<FMT>`

The original plan included phases for `Model.fmt_model`,
`Functional<FMT>` class template, and internalizing weight types.
After completing the core method migration: the FMT model is a
**method choice** (not a model parameter), so it stays separate from
`physics::Model`; templatizing `Functional` adds complexity for a
marginal `std::visit` gain on the hot path; and internalizing weights
depends on the template class. These remain available as future
optimizations if profiling identifies the `std::visit` as a bottleneck.

---

## 3. Progress log

| Phase | Status | Tests after |
|-------|--------|-------------|
| Phase 1 (Potentials) | ✅ Complete | 492 unit (9744 asserts), 59 integration (3846 asserts) |
| Phase 2 (HS + EoS) | ✅ Complete | 492 unit (9744 asserts), 59 integration (3846 asserts) |
| Phase 3 (FMT models) | ✅ Complete | 492 unit (9744 asserts), 59 integration (3846 asserts) |
| Phase 4 (Geometry + Grid) | ✅ Complete | 492 unit (9744 asserts), 59 integration (3846 asserts) |
| Phase 5 (Orphan audit + bulk) | ✅ Complete | 492 unit (9744 asserts), 59 integration (3846 asserts) |

### Summary of changes

**Files modified (methods added to structs):**
- `include/dft/physics/potentials.hpp` — `LennardJones`, `TenWoldeFrenkel`, `WangRamirezDobnikarFrenkel`: `operator()`, `from_r2()`, `energy()`, `repulsive()`, `attractive()`, `hard_core_diameter()`, `split_point()`
- `include/dft/physics/hard_spheres.hpp` — `CarnahanStarling`, `PercusYevickVirial`, `PercusYevickCompressibility`: `d_excess_free_energy()`, `d2_excess_free_energy()`, `d3_excess_free_energy()`, `pressure()`, `free_energy()`, `chemical_potential()`
- `include/dft/physics/eos.hpp` — `IdealGas`, `PercusYevick`, `LennardJonesJZG`, `LennardJonesMecke`: `d_excess_free_energy()`, `d2_excess_free_energy()`, `free_energy()`, `excess_chemical_potential()`, `chemical_potential()`, `pressure()`
- `include/dft/functionals/fmt/models.hpp` — `Rosenfeld`, `RSLT`, `WhiteBearI`, `WhiteBearII`, `EsFMT`: `d_f1()`, `d_f2()`, `d_f3()`, `phi()`, `d_phi()`, `free_energy_density()`, `excess_chemical_potential()`; common logic in `detail::compute_phi<>`, `detail::compute_d_phi<>`, `detail::compute_excess_chemical_potential<>`
- `include/dft/geometry/vertex.hpp` — `Vertex::dimension()`
- `include/dft/geometry/element.hpp` — `SquareBox2D::volume()`, `SquareBox3D::volume()`
- `include/dft/geometry/mesh.hpp` — `UniformMesh2D/3D::volume()`, `element_volume()`, `spacing()`
- `include/dft/grid.hpp` — `Grid::for_each_wavevector()`; `Wavevector` moved before `Grid`
- `include/dft/functionals/bulk/phase_diagram.hpp` — `PhaseDiagram::interpolate()`

**All original free functions retained as thin wrappers** for backward compatibility.
**Zero test files modified.** All 492 unit tests (9744 assertions) and 59 integration tests (3846 assertions) pass.
