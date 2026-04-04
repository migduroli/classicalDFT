# Nucleation: critical droplet and barrier crossing

## Physical background

Classical nucleation theory (CNT) describes the formation of a new phase
(liquid) from a metastable parent phase (supersaturated vapor). The free
energy landscape contains a saddle point corresponding to the critical
nucleus: a density profile that is in unstable equilibrium with the vapor.
Droplets smaller than the critical size dissolve; larger ones grow.

This example performs the complete nucleation analysis within classical DFT:
finding the critical cluster, confirming it is a saddle point, and simulating
the post-critical dynamics in both directions (dissolution and growth).

### The grand potential functional

At fixed chemical potential $\mu$ and temperature $T$, the equilibrium
density profile minimises the grand potential:

$$
\Omega[\rho] = F[\rho] - \mu\int\rho(\mathbf{r})\,d\mathbf{r}
$$

where $F[\rho] = F_{\mathrm{id}} + F_{\mathrm{HS}} + F_{\mathrm{mf}}$ is the
Helmholtz free energy functional (see the density example for details).

For a supersaturated vapor ($\mu > \mu_{\mathrm{coex}}$), the uniform vapor is
a local minimum of $\Omega$. The critical cluster is a saddle point: a
stationary point of $\Omega[\rho]$ with exactly one negative eigenvalue of
the Hessian.

### Phase 1: finding the critical cluster (FIRE with mass constraint)

The critical cluster is found by minimising the Helmholtz free energy $F$ at
fixed total mass $M = \int\rho\,d\mathbf{r}$, starting from a step-function
initial condition:

$$
\rho_0(\mathbf{r}) = \begin{cases}
\rho_l, & |\mathbf{r} - \mathbf{r}_c| < R_0 \\
\rho_{\mathrm{out}}, & \text{otherwise}
\end{cases}
$$

where $\rho_l$ is the liquid coexistence density, $\rho_{\mathrm{out}}$ is the
background (supersaturated vapor) density, and $R_0$ is the initial droplet radius.

The minimisation uses the FIRE2 algorithm with a Lagrange multiplier
$\lambda$ to enforce the mass constraint. The alias parametrisation
$x_i = \sqrt{\max(0, \rho_i - \rho_{\min})}$, $\rho_i = \rho_{\min} + x_i^2$
ensures positivity of the density.

The constrained force is:

$$
\frac{\partial\mathcal{L}}{\partial x_i} = 2x_i\left(\frac{\delta F}{\delta\rho_i} - \lambda\,\Delta V\right)
$$

where $\lambda = \sum_i (\delta F/\delta\rho_i)\,\rho_i\,/\,M$ is chosen so that the
projected force conserves mass.

Homogeneous boundary conditions are applied: forces on boundary (face) grid
points are averaged, preventing artificial gradients at the periodic
boundaries.

### Phase 2: eigenvalue analysis (saddle point confirmation)

The critical cluster should be a saddle point of $\Omega[\rho]$ with exactly
one unstable direction. The smallest eigenvalue of the Hessian
$\hat{H} = \delta^2\Omega/\delta\rho^2$ is computed using a Rayleigh-quotient
FIRE2 algorithm.

The Hessian-vector product is approximated by finite differences:

$$
\hat{H}\mathbf{v} \approx \frac{\nabla\Omega(\rho + \epsilon\mathbf{v}) - \nabla\Omega(\rho)}{\epsilon}
$$

The Rayleigh quotient $R(\mathbf{v}) = \mathbf{v}^T\hat{H}\mathbf{v}/|\mathbf{v}|^2$
is minimised over unit vectors $\mathbf{v}$ using FIRE2 applied to the
objective:

$$
f(\mathbf{v}) = R(\mathbf{v}) + (|\mathbf{v}|^2 - 1)^2
$$

The penalty term ensures normalisation. A negative eigenvalue confirms the
saddle point character.

### Phase 3: DDFT dynamics (dissolution and growth)

The eigenvector $\mathbf{e}$ corresponding to the negative eigenvalue points
along the unstable direction. Perturbing the critical cluster density along
$\pm\mathbf{e}$ and evolving with DDFT reveals the two fates:

- $\rho - s\,\mathbf{e}$: the droplet dissolves back to the uniform vapor.
- $\rho + s\,\mathbf{e}$: the droplet grows toward the stable liquid phase.

The DDFT integrating-factor scheme is used (same as in the dynamics example).
The nucleation barrier height is:

$$
\Delta\Omega = \Omega[\rho_{\mathrm{cluster}}] - \Omega[\rho_{\mathrm{vapor}}]
$$

### Model parameters

The example uses a Lennard-Jones fluid with:

| Parameter | Value |
|-----------|-------|
| $\sigma$ | 1.0 |
| $\varepsilon$ | 1.0 |
| $r_c$ | 3.0 |
| $T^* = k_BT/\varepsilon$ | 0.8 |
| $\Delta x$ | 0.4$\sigma$ |
| Box | $12.8^3\sigma^3$ (32^3 grid) |
| FMT model | RSLT |
| Splitting | WCA |
| Weight scheme | InterpolationQuadraticF (27-point) |

These parameters match Jim's `Droplet/LJ/M103/T0.8/input.dat` configuration,
except for the grid spacing ($\Delta x = 0.4$ vs Jim's $\Delta x = 0.1$).
The coarser grid is used for speed ($32^3$ vs $128^3$ points).

## What the code does

1. Sets up the LJ model with RSLT FMT and 27-point interaction quadrature.
2. Finds liquid-vapor coexistence and adjusts the supersaturation.
3. Runs FIRE2 with mass constraint to find the critical cluster.
4. Computes the nucleation barrier height $\Delta\Omega$.
5. Computes the smallest eigenvalue of the Hessian at the cluster to confirm
   the saddle point.
6. Perturbs along the eigenvector and runs DDFT in both directions
   (dissolution and growth), logging $\Omega(t)$ and effective radius $R(t)$.

## Cross-validation (`check/`)

The check program performs a systematic step-by-step comparison against Jim's
classicalDFT library:

| Step | Category | Quantities | Reference | Tolerance |
|------|----------|-----------|-----------|-----------|
| 1 | LJ potential | $v(r)$, shift, $r_{\min}$, $V_{\min}$, $r_0$, $w_{\mathrm{att}}(r)$ | Jim's `Potential1.h` | $10^{-10}$ |
| 2 | Hard-sphere diameter | $d_{\mathrm{HS}}$ (BH integral) | Jim's `Species.h` | $10^{-10}$ |
| 3 | Analytical $a_{\mathrm{vdw}}$ | Continuum integral | Jim's `Species.h` | $10^{-6}$ |
| 4 | Grid $a_{\mathrm{vdw}}$ | 27-point QF quadrature | Jim's `Interaction.cpp` | $10^{-6}$ |
| 5 | Bulk thermo | $\mu(\rho_{\mathrm{out}})$, coexistence, supersaturation | Jim's `EOS.h` | $10^{-6}$ |
| 6 | FIRE minimisation | $\max|\rho_{\mathrm{ours}} - \rho_{\mathrm{Jim}}|$ | Jim's FIRE2 | $10^{-2}$ |
| 6b | Barrier height | $\Delta\Omega_{\mathrm{ours}} \approx \Delta\Omega_{\mathrm{Jim}}$ | Jim's FIRE2 | $10\%$ relative |
| 7 | Eigenvalue | Smallest $\hat{H}$ eigenvalue; $|\mathbf{e}\cdot\mathbf{e}_{\mathrm{Jim}}|$ | Jim's eigenvalue solver | $10\%$; $> 0.9$ |
| 8+ | DDFT dynamics | $\max|\rho(t)_{\mathrm{ours}} - \rho(t)_{\mathrm{Jim}}|$ per step | Jim's DDFT | $10^{-3}$ per step |

## Build and run

```bash
make run-local  # local build (runs the full nucleation workflow)
make run-checks # cross-validation against Jim's code
```

## Output

### Droplet evolution

Cross-sectional density profile through the droplet centre. The initial
step-function seed evolves toward a smooth equilibrium profile.

![Droplet evolution](exports/droplet_evolution.png)

### Grand potential

The grand potential $\Omega$ during DDFT dynamics. Dissolution (blue) drives
$\Omega$ down to the uniform vapor value; growth (red) drives $\Omega$ down
toward the liquid minimum.

![Grand potential](exports/grand_potential.png)
