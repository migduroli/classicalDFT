# Dynamics: minimisation and time evolution

## Physical background

This example demonstrates two algorithmic building blocks used throughout the
library: the FIRE2 minimiser for finding free energy minima, and the
split-operator DDFT scheme for time evolution.

### FIRE2 minimiser

The Fast Inertial Relaxation Engine (FIRE) is a molecular dynamics-inspired
optimisation algorithm. Starting from an initial configuration $\mathbf{x}_0$
with zero velocity, it integrates the equation of motion:

$$
m\ddot{\mathbf{x}} = -\nabla E(\mathbf{x})
$$

with an adaptive time step and a velocity mixing rule that biases the
direction of motion toward the force direction:

$$
\mathbf{v} \leftarrow (1 - \alpha)\mathbf{v} + \alpha\,\frac{|\mathbf{v}|}{|\mathbf{F}|}\,\mathbf{F}
$$

The mixing parameter $\alpha$ starts large (strong damping) and decays as the
system accumulates downhill power $P = \mathbf{F}\cdot\mathbf{v} > 0$. When
$P < 0$ (uphill motion), the time step is halved, velocity is zeroed, and
$\alpha$ is reset.

The FIRE2 variant uses additional criteria:
- $\Delta t$ increases only after $N_{\mathrm{delay}}$ consecutive positive-power steps.
- $\alpha$ decays multiplicatively: $\alpha \leftarrow f_\alpha \cdot \alpha$.
- Convergence is declared when the RMS force falls below tolerance.

The example minimises a 2D quadratic $f(x,y) = (x-1)^2 + 4(y+2)^2$ from
the initial guess $(5, 5)$.

### Split-operator DDFT

The DDFT equation of motion for a conserved density field is:

$$
\frac{\partial\rho}{\partial t} = D\,\nabla\cdot\left[\rho\,\nabla\frac{\delta F}{\delta\rho}\right]
$$

For an ideal gas, $\delta F/\delta\rho = k_BT\ln\rho$, and the equation
reduces to the diffusion equation:

$$
\frac{\partial\rho}{\partial t} = D\,\nabla^2\rho
$$

The split-operator scheme separates the ideal-gas (linear) term from the
excess (nonlinear) contribution. The ideal-gas propagator is applied exactly
in Fourier space:

$$
\hat{\rho}_k(t + \Delta t) = e^{\Lambda_k D\Delta t}\,\hat{\rho}_k(t)
$$

where $\Lambda_k = (2/\Delta x^2)(\cos k_i - 1)$ is the eigenvalue of the
discrete Laplacian at wavenumber $k$. The excess contribution is added via
a second-order integrating factor correction.

For the ideal gas test case, $\rho(\mathbf{r}, 0) = \rho_0 + A\sin(2\pi z/L)$.
The amplitude of the fundamental mode decays as:

$$
A(t) = A(0)\, e^{\Lambda_1 D t}
$$

This provides an exact analytical check: the numerical decay rate must match
$e^{\Lambda_1 D\Delta t}$ per step.

---

## Key library types

| Type | Header | Role |
|------|--------|------|
| `algorithms::fire::Fire` | `dft/algorithms/fire.hpp` | FIRE2 minimiser configuration (timestep, mixing, tolerance) |
| `Grid` | `dft/grid.hpp` | Spatial discretisation and FFT plans |
| `physics::Model` | `dft/physics/model.hpp` | Aggregate of `Grid` + species + interactions + temperature |
| `algorithms::dynamics::Simulation` | `dft/algorithms/dynamics.hpp` | Split-operator DDFT integrator with adaptive timestep |
| `algorithms::dynamics::StepConfig` | `dft/algorithms/dynamics.hpp` | Per-step settings: `dt`, `diffusion_coefficient`, `min_density`, `dt_max` |

`Fire` is configured via designated initializers and called with `.run(f, x0)`
where `f` returns `(energy, gradient)`. `Simulation` wraps the DDFT
integrating-factor scheme and is called with `.run(rho0, grid, force_fn)`.

---

## Step-by-step code walkthrough

### Part A: FIRE2 minimisation of a 2D quadratic

#### Step 1: Read FIRE2 parameters

All parameters are loaded from `config.ini`:

```cpp
auto cfg = config::parse_config("config.ini", config::FileType::INI);
double fire_dt = config::get<double>(cfg, "fire2.dt");
double fire_dt_max = config::get<double>(cfg, "fire2.dt_max");
double fire_tol = config::get<double>(cfg, "fire2.force_tolerance");
int fire_max_steps = config::get<int>(cfg, "fire2.max_steps");
double fire_x0 = config::get<double>(cfg, "fire2.x0");
double fire_y0 = config::get<double>(cfg, "fire2.y0");
```

#### Step 2: Define the objective and force

The test function is a 2D anisotropic quadratic:

$$
f(x, y) = (x - 1)^2 + 4(y + 2)^2
$$

with analytical gradient $\nabla f = (2(x-1),\; 8(y+2))$. The force is
$\mathbf{F} = -\nabla f$:

```cpp
auto fire_force_fn = [](const std::vector<arma::vec>& x)
    -> std::pair<double, std::vector<arma::vec>> {
  double xv = x[0](0);
  double yv = x[0](1);
  double energy = (xv - 1.0) * (xv - 1.0) + 4.0 * (yv + 2.0) * (yv + 2.0);
  arma::vec force = {-2.0 * (xv - 1.0), -8.0 * (yv + 2.0)};
  return {energy, {force}};
};
```

The force callback follows the library convention: it takes a vector of
`arma::vec` (one per species/degree-of-freedom) and returns `{energy, forces}`.

#### Step 3: One-shot minimisation

The `Fire::minimize()` method runs FIRE2 to convergence:

```cpp
algorithms::fire::Fire fire_config{
    .dt = fire_dt, .dt_max = fire_dt_max,
    .force_tolerance = fire_tol, .max_steps = fire_max_steps,
};
auto fire_result = fire_config.minimize({arma::vec{fire_x0, fire_y0}}, fire_force_fn);
```

This returns `{x, energy, rms_force, converged, iteration}`. The solution
should be $(1, -2)$ with energy $= 0$.

#### Step 4: Step-by-step iteration

For convergence logging, the same minimisation is run using the
`initialize()` + `step()` API:

```cpp
auto state = fire_config.initialize({arma::vec{fire_x0, fire_y0}}, fire_force_fn);
auto [init_energy, forces] = fire_force_fn(state.x);

while (!state.converged && state.iteration < fire_config.max_steps) {
  auto [new_state, new_forces] = fire_config.step(std::move(state), forces, fire_force_fn);
  state = std::move(new_state);
  forces = std::move(new_forces);
}
```

This loop records energy values at configurable intervals, producing the
convergence plot below.

---

### Part B: Split-operator DDFT (ideal gas)

#### Step 5: Define the ideal gas model

A zero-diameter species with no interactions defines a pure ideal gas:

```cpp
physics::Model model{
    .grid = make_grid(ddft_dx, {box_size, box_size, box_size}),
    .species = {Species{.name = "ideal", .hard_sphere_diameter = 0.0}},
    .interactions = {},
    .temperature = temperature,
};
```

#### Step 6: Construct the sinusoidal initial condition

A sinusoidal perturbation along the $z$-axis is replicated over $x$ and $y$:

```cpp
arma::vec z_vals = arma::linspace(0.0, (nz - 1) * model.grid.dx, nz);
arma::vec z_profile = rho0 + amplitude * arma::sin(
    2.0 * std::numbers::pi * z_vals / model.grid.box_size[2]);
arma::vec rho = arma::repmat(z_profile, nx * ny, 1);
```

The density varies only in $z$, making the decay rate analytically
predictable.

#### Step 7: Define the ideal-gas force function

For an ideal gas, $F[\rho] = k_BT \int \rho(\ln\rho - 1)\,d\mathbf{r}$
and $\delta F/\delta\rho = k_BT \ln\rho$. Since $k_BT = 1$:

```cpp
auto ddft_force_fn = [&](const std::vector<arma::vec>& densities)
    -> std::pair<double, std::vector<arma::vec>> {
  double dv = model.grid.cell_volume();
  arma::vec rho_safe = arma::clamp(densities[0], 1e-18, arma::datum::inf);
  double energy = dv * arma::dot(rho_safe, arma::log(rho_safe) - 1.0);
  arma::vec force = dv * arma::log(rho_safe);
  return {energy, {force}};
};
```

#### Step 8: Run the DDFT simulation

The `Simulation` struct configures and runs the time integration:

```cpp
algorithms::dynamics::Simulation sim_config{
    .step = {.dt = ddft_dt, .diffusion_coefficient = D, .min_density = 1e-18},
    .n_steps = n_steps,
    .snapshot_interval = snapshot_interval,
    .log_interval = snapshot_interval,
};

auto sim = sim_config.run(
    {ddft_state.species[0].density.values}, model.grid, ddft_force_fn);
```

The simulation stores density snapshots at regular intervals, allowing
post-hoc analysis of the density evolution and variance decay.

#### Step 9: Verify variance decay and mass conservation

The variance of the density field should decay exponentially:

$$
\mathrm{Var}[\rho](t) \propto e^{2\Lambda_1 D t}
$$

and the total mass must be conserved to machine precision:

```cpp
std::println(std::cout, "  Mass initial: {:.6f}", sim.mass_initial);
std::println(std::cout, "  Mass final:   {:.6f}", sim.mass_final);
std::println(std::cout, "  Rel. error:   {:.6e}",
             std::abs(sim.mass_final - sim.mass_initial) / sim.mass_initial);
```

---

## Cross-validation (`check/`)

| Step | Test | Analytical reference | Tolerance |
|------|------|---------------------|-----------|
| 1 | Propagator coefficients | $e^{\Lambda_k D\Delta t}$ at $k=0$ (= 1), $k=1$, Nyquist | $10^{-10}$ |
| 2 | Pure diffusion step | Amplitude decay $A(t)/A(0) = e^{\Lambda_1 D t}$; mass $= M_0$ | $10^{-4}$ (decay); $10^{-10}$ (mass) |
| 3 | Full DFT step | Mass conservation; $\Omega(t+\Delta t) \leq \Omega(t)$ | $10^{-6}$ (mass); monotone |

Step 2 uses an ideal gas with $\rho(\mathbf{r},0) = \rho_0 + A\cos(2\pi z/L)$
and verifies the analytical exponential decay rate. Step 3 uses a tanh slab
at $kT = 0.7$ for an LJ fluid and verifies that DDFT respects both mass
conservation and the second law (monotonic decrease of $\Omega$).

## Build and run

```bash
make run        # Docker
make run-local  # local build
make run-checks # cross-validation
```

## Output

### FIRE2 energy convergence

Energy decays rapidly from the initial guess $(5, 5)$ toward the minimum at
$(1, -2)$.

![FIRE2 energy](exports/fire2_energy.png)

### DDFT density variance decay

The variance of the density field decays exponentially as the sinusoidal
perturbation relaxes to the uniform equilibrium.

![DDFT variance](exports/ddft_variance.png)

### DDFT density profile evolution

Snapshots of the 1D density profile $\rho(z)$ at successive times.

![DDFT profiles](exports/ddft_density_profiles.png)
