# Dynamics

## Overview

The `dft::dynamics` namespace provides minimisers and time integrators
for density functional theory calculations.

| Class | Role |
|-------|------|
| `Minimizer` | Abstract base class (NVI pattern) for all minimisers and time-steppers |
| `Fire2Minimizer` | FIRE2 algorithm: velocity-Verlet with adaptive timestep and velocity damping |
| `Integrator` | Density dynamics integrator with configurable scheme |
| `IntegrationScheme::SplitOperator` | Exact diffusion (FFT) + explicit Euler for excess |
| `IntegrationScheme::CrankNicholson` | Exact diffusion (FFT) + implicit Crank-Nicholson for excess |

## Usage

### FIRE2 minimisation

```cpp
#include "dft.h"

using namespace dft;

// Create a solver with one species at rho = 0.3
auto solver = make_solver(...);
solver.species(0).set_chemical_potential(std::log(0.7));

// Configure and run FIRE2
dynamics::Fire2Config config{.dt = 1e-3, .dt_max = 0.1, .force_limit = 1e-8};
dynamics::Fire2Minimizer fire(solver, config);
bool converged = fire.run(500);
```

### Density dynamics (split-operator)

```cpp
dynamics::IntegratorConfig config{
    .scheme = dynamics::IntegrationScheme::SplitOperator,
    .dt = 1e-4,
    .diffusion_coefficient = 1.0,
    .force_limit = 1e-12,
};
dynamics::Integrator integrator(solver, config);
integrator.run(100);
```

### Density dynamics (Crank-Nicholson)

```cpp
dynamics::IntegratorConfig config{
    .scheme = dynamics::IntegrationScheme::CrankNicholson,
    .dt = 5e-4,
    .diffusion_coefficient = 1.0,
    .force_limit = 1e-12,
    .crank_nicholson_iterations = 5,
    .cn_tolerance = 1e-10,
};
dynamics::Integrator integrator(solver, config);
integrator.run(100);
```

### Step callbacks

All minimisers support step callbacks for monitoring convergence:

```cpp
fire.set_step_callback([](long step, double energy, double max_force) {
    std::cout << "Step " << step << ": E = " << energy << "\n";
    return true;  // return false to stop iteration
});
```

## Running

```bash
make run        # builds and runs inside Docker
make run-local  # builds and runs locally
```

## Plots

When built with matplotlib support (`DFT_USE_MATPLOTLIB=ON`), the example
produces the following plots in `exports/`:

| File | Description |
|------|-------------|
| `fire2_energy.png` | FIRE2 energy convergence |
| `fire2_force.png` | FIRE2 force convergence |
| `split_operator_dynamics.png` | Density profiles at successive times (split-operator) |
| `crank_nicholson_dynamics.png` | Density profiles at successive times (Crank-Nicholson) |
| `scheme_comparison.png` | Variance decay comparison of both integration schemes |
