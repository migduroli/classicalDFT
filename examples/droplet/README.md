# Droplet nucleation

Demonstrates liquid droplet formation and the planar liquid-vapor interface
using mean-field DFT with a Lennard-Jones potential.

## What it does

**Demo 1 — Planar interface:** Constructs a liquid slab in a 3D periodic box
($6\sigma \times 6\sigma \times 12\sigma$) at $T^* = 0.80$ with the chemical
potential set to coexistence. FIRE2 relaxes the initial tanh profile towards
the self-consistent interface shape.

**Demo 2 — Sub-critical droplet:** Seeds a spherical liquid droplet (radius
$\approx 1.5\sigma$) inside a slightly supersaturated vapor ($\mu = \mu_{\text{coex}} + 0.01$)
in an $8\sigma$ cubic box. Because $R < R_{\text{critical}}$, the FIRE2
minimiser dissolves the droplet into uniform vapor, confirming that the
system sits below the nucleation barrier.

## Physics

At temperatures below the critical point, a Lennard-Jones fluid exhibits
liquid-vapor coexistence. The coexistence densities are found via
`Solver::find_coexistence()`.

In demo 1, setting $\mu = \mu_{\text{coex}}$ makes the liquid and vapor
phases equally stable. The planar interface is a stationary point of the
grand potential functional. FIRE2 relaxes the interface shape while the slab
position remains fixed by symmetry.

In demo 2, the chemical potential is raised slightly above coexistence
(supersaturation). Droplets smaller than the critical nucleation radius
$R_{\text{critical}}$ are unstable: the surface energy cost exceeds the
volume free-energy gain. The minimiser correctly finds the uniform vapor as
the lower-energy state, and the difference
$\Delta\Omega = \Omega[\rho_{\text{final}}] - \Omega[\rho_v \cdot V]$
quantifies the free-energy cost of the initial seed.

## Key API usage

```cpp
// Build solver with FMT + LJ interaction
Solver solver;
auto sp = std::make_unique<functional::fmt::Species>(std::move(dens), diameter);
auto& sp_ref = *sp;
solver.add_species(std::move(sp));
solver.add_interaction(
    std::make_unique<functional::interaction::Interaction>(sp_ref, sp_ref, lj, kT));
solver.set_fmt(std::make_unique<functional::fmt::FMT>(functional::fmt::WhiteBearII{}));

// Find coexistence and set chemical potential
double rho_v, rho_l;
solver.find_coexistence(1.1, 0.005, rho_v, rho_l, 1e-8);
solver.species(0).set_chemical_potential(solver.chemical_potential(rho_v) + delta_mu);

// Initialise droplet (tanh profile) and minimise
init_droplet(solver, rho_v, rho_l, r_droplet, interface_width);
dynamics::Fire2Config config{.dt = 1e-3, .dt_max = 0.01, .force_limit = 5e-3};
dynamics::Fire2Minimizer fire(solver, config);
fire.run(500);
```

## Running

```bash
make run-local    # build and run locally
make run          # build and run in Docker
```

## Output

| Plot | File | Description |
|------|------|-------------|
| Planar interface | `exports/planar_interface.png` | Density profile $\rho(z)$ of a liquid slab |
| Droplet profile | `exports/droplet_profile.png` | Radial $\rho(r)$: initial seed vs minimised (dissolved) |
| Convergence | `exports/droplet_convergence.png` | FIRE2 energy vs step for the droplet demo |
