# Density: DFT functional evaluation and DDFT relaxation

## Physical background

This is the flagship example of the library. It demonstrates the complete
classical DFT workflow: defining a physical model, evaluating the full
inhomogeneous free energy functional via FFT convolutions, and relaxing an
initial density profile toward equilibrium via DDFT.

### The DFT free energy functional

The total Helmholtz free energy functional for a one-component fluid is:

$$
F[\rho] = F_{\mathrm{id}}[\rho] + F_{\mathrm{HS}}[\rho] + F_{\mathrm{mf}}[\rho]
$$

**Ideal gas**:

$$
F_{\mathrm{id}}[\rho] = k_BT \int \rho(\mathbf{r})\left[\ln\rho(\mathbf{r}) - 1\right] d\mathbf{r}
$$

**Hard-sphere (FMT)**:

$$
F_{\mathrm{HS}}[\rho] = k_BT \int \Phi\bigl(\{n_\alpha(\mathbf{r})\}\bigr)\, d\mathbf{r}
$$

where $\{n_\alpha\}$ are the FMT weighted densities obtained by convolving
$\rho$ with the weight functions (see the FMT example).

**Mean-field attraction**:

$$
F_{\mathrm{mf}}[\rho] = \frac{1}{2}\int\!\int \rho(\mathbf{r})\, w_{\mathrm{att}}(|\mathbf{r}-\mathbf{r}'|)\, \rho(\mathbf{r}')\, d\mathbf{r}\, d\mathbf{r}'
$$

On the periodic grid, all convolutions are computed via FFT:
$n_\alpha = \mathrm{IFFT}[\hat{\rho}\cdot\hat{w}_\alpha]$. This reduces the
cost from $O(N^2)$ to $O(N\log N)$ per functional evaluation.

### The grand potential

At fixed chemical potential $\mu$, the equilibrium density minimises the
grand potential:

$$
\Omega[\rho] = F[\rho] - \mu \int \rho(\mathbf{r})\, d\mathbf{r}
$$

The forces (functional derivatives) are:

$$
\frac{\delta\Omega}{\delta\rho(\mathbf{r})} = \frac{\delta F}{\delta\rho(\mathbf{r})} - \mu
$$

At equilibrium all forces vanish. A non-zero force drives the density toward
lower $\Omega$.

### Bulk-inhomogeneous consistency

A stringent test of the FFT convolution pipeline is that a uniform density
profile must reproduce bulk thermodynamics exactly:

$$
F[\rho = \mathrm{const}] = f_{\mathrm{bulk}}(\rho) \times V
$$

where $f_{\mathrm{bulk}}$ is the bulk free energy density from the
thermodynamics module. This identity verifies the entire chain: weight
generation, FFT convolution, FMT $\Phi$ evaluation, mean-field energy
accumulation, and force back-convolution.

### DDFT dynamics

The density evolves according to the DDFT equation of motion (conserved dynamics):

$$
\frac{\partial\rho}{\partial t} = D\,\nabla\cdot\left[\rho\,\nabla\frac{\delta\Omega}{\delta\rho}\right]
$$

where $D$ is the diffusion coefficient. The split-operator scheme separates
the ideal-gas (linear) part from the excess (nonlinear) part and propagates
the linear part exactly in Fourier space using the integrating factor
$\exp(\Lambda_k D\Delta t)$, where $\Lambda_k$ are the discrete Laplacian
eigenvalues.

### Liquid slab geometry

The initial condition is a planar liquid slab constructed as a symmetric
tanh profile:

$$
\rho(x) = \rho_v + \frac{\rho_l - \rho_v}{2}\left[\tanh\!\left(\frac{x - x_c + w}{\xi}\right) - \tanh\!\left(\frac{x - x_c - w}{\xi}\right)\right]
$$

where $\rho_v$ and $\rho_l$ are the coexisting vapor and liquid densities,
$x_c$ is the box centre, $w$ is the half-width, and $\xi$ is the interface
width. DDFT relaxation sharpens the interface toward the equilibrium DFT
solution.

## What the code does

1. Defines an LJ fluid at $T^* = 0.7$ on a $40\times20\times20$ grid
   ($\Delta x = 0.25\sigma$) with WCA splitting and White Bear II FMT.
2. Evaluates bulk $f(\rho)$, $\mu(\rho)$, $P(\rho)$ and finds coexistence.
3. Constructs a tanh liquid slab profile.
4. Evaluates the full DFT functional (free energy, grand potential, forces).
5. Runs 500 DDFT split-operator steps. The grand potential decreases
   monotonically and mass is conserved to machine precision.

## Cross-validation (`check/`)

The check program validates the FFT convolution pipeline by requiring that
inhomogeneous DFT at uniform density reproduces bulk thermodynamics exactly.

| Step | Test | Quantity | Grid | Tolerance |
|------|------|---------|------|-----------|
| 1-4 | $F_{\mathrm{inhom}} = f_{\mathrm{bulk}} \times V$ | $F_{\mathrm{id}}, F_{\mathrm{HS}}, F_{\mathrm{mf}}, F_{\mathrm{total}}$ | $\rho = 0.1, 0.3, 0.5, 0.7$ | $10^{-6}$ relative |
| 5 | Zero force at equilibrium | $\max|\delta\Omega/\delta\rho|$ at $\rho = 0.4$, $\mu = \mu_{\mathrm{bulk}}$ | $12^3$ grid | $10^{-6}$ |
| 6 | $\Omega/V = -P_{\mathrm{coex}}$ | Grand potential at both coexistence densities | $\rho_v, \rho_l$ | $10^{-6}$ relative |

Steps 1-4 break down the free energy into $F_{\mathrm{id}}$, $F_{\mathrm{HS}}$,
and $F_{\mathrm{mf}}$ individually, so any discrepancy can be localised to a
specific contribution. Step 5 validates the entire derivative chain: FMT
back-convolution, mean-field force, and ideal gas force. Step 6 confirms
the grand potential identity at phase coexistence.

## Build and run

```bash
make run        # Docker
make run-local  # local build
make run-checks # cross-validation
```

## Output

### Pressure isotherm

The van der Waals loop in $P(\rho)$, with the Maxwell construction tie-line
connecting the coexisting phases.

![Pressure isotherm](exports/pressure_isotherm.png)

### Free energy density

The double-well structure of $f(\rho)$ at sub-critical temperature, with the
common tangent construction.

![Free energy density](exports/free_energy.png)

### Chemical potential

$\mu(\rho)$ with the non-monotonic (unstable) region between the spinodal
densities.

![Chemical potential](exports/chemical_potential.png)

### DDFT density evolution

The initial tanh slab profile relaxes toward the equilibrium DFT solution.

![Density evolution](exports/density_evolution.png)

### Grand potential convergence

The grand potential $\Omega$ decreases monotonically during DDFT relaxation,
confirming the thermodynamic consistency of the dynamics.

![Grand potential](exports/grand_potential.png)
