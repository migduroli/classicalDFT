# Droplet nucleation

Demonstrates classical nucleation theory from first principles using
mean-field DFT with a Lennard-Jones fluid: locating the critical nucleus,
and showing DDFT dynamics of sub-critical (dissolution) and super-critical
(growth) droplets.

## What it does

1. **Nucleation barrier scan.** At $T^* = 1.0$ and supersaturation
   $\Delta\mu = 0.30$, evaluates the grand potential $\Omega[\rho_R]$ for
   tanh-profile droplets of varying radius $R$. The maximum of
   $\Delta\Omega(R) = \Omega[\rho_R] - \Omega[\rho_v]$ locates the critical
   nucleus at $R^* \approx 2\sigma$ with a barrier $\approx 1.5\,k_BT$.

2. **Dissolution dynamics.** Initialises a sub-critical droplet ($R = R^* -
   0.5\sigma$) and evolves it with the DDFT integrator (split-operator
   scheme). Snapshots show the radial density profile collapsing towards
   uniform vapor.

3. **Growth dynamics (small overshoot).** Initialises a super-critical
   droplet ($R = R^* + 0.5\sigma$). The DDFT evolution shows the droplet
   expanding as the system rolls downhill past the barrier.

4. **Growth dynamics (large overshoot).** Same as above with $R = R^* +
   1.0\sigma$, showing faster growth from a larger initial droplet.

## Physics

At $\mu > \mu_{\text{coex}}$, the liquid phase is thermodynamically favoured.
Forming a droplet of radius $R$ involves a volume free-energy gain
$\propto -R^3$ and a surface energy cost $\propto R^2$. Their competition
produces a free-energy barrier with a maximum at the critical radius
$R^*$. Droplets with $R < R^*$ dissolve; droplets with $R > R^*$ grow.

The DDFT equation

$$\frac{\partial \rho}{\partial t} = D\,\nabla \cdot \left[\rho\,\nabla \frac{\delta \beta F}{\delta \rho}\right]$$

drives the density down the grand-potential landscape.

## Key API usage

```cpp
// Evaluate grand potential for a given droplet radius.
init_droplet(solver, rho_v, rho_l, R, interface_width);
double omega = solver.compute_free_energy_and_forces();

// Run DDFT dynamics and capture snapshots.
dynamics::IntegratorConfig iconf{
    .scheme = dynamics::IntegrationScheme::SplitOperator,
    .dt = 5e-4,
    .diffusion_coefficient = 1.0,
    .force_limit = 1e-12,
};
dynamics::Integrator integrator(solver, iconf);
for (int s = 0; s < n_snaps; ++s) {
  (void)integrator.resume(steps_per_snap);
  // ... extract radial profile snapshot ...
}
```

## Running

```bash
make run-local    # build and run locally
make run          # build and run in Docker
```

## Output

| Plot | File | Description |
|------|------|-------------|
| Nucleation barrier | `exports/nucleation_barrier.png` | $\Delta\Omega(R)$ curve with $R^*$ marked |
| Dissolution | `exports/dissolution.png` | Sub-critical droplet dissolving (blue snapshots) |
| Growth (small) | `exports/growth.png` | Super-critical droplet expanding, $R_0 = R^* + 0.5\sigma$ (red snapshots) |
| Growth (large) | `exports/growth_large.png` | Larger super-critical droplet, $R_0 = R^* + 1.0\sigma$ (orange snapshots) |
