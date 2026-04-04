# Picard: self-consistent density equilibration

## Purpose

This doc demonstrates Picard (self-consistent field) iteration for finding
the equilibrium density profile that minimises the grand potential. Picard
iteration is the simplest density-functional minimizer: at each step, the
new density is obtained by exponentiating the current force field. It
complements the FIRE minimizer (used in the nucleation doc) and the DDFT
integrator (used in the dynamics and density docs).

## Mathematical background

### The Euler-Lagrange equation

At equilibrium, the density profile $\rho(\mathbf{r})$ satisfies:

$$
\frac{\delta\Omega}{\delta\rho(\mathbf{r})} = 0
$$

Expanding the grand potential $\Omega = F[\rho] - \mu N$ into ideal and
excess parts:

$$
\ln\rho(\mathbf{r}) + \frac{\delta F_{\mathrm{ex}}}{\delta\rho(\mathbf{r})} - \beta\mu = 0
$$

Rearranging gives the self-consistency relation:

$$
\rho(\mathbf{r}) = \exp\!\left(\beta\mu - \frac{\delta F_{\mathrm{ex}}}{\delta\rho(\mathbf{r})}\right)
$$

### Log-space Picard iteration

Direct substitution of the self-consistency relation is unstable. The
library uses a damped log-space update:

$$
\rho_{n+1}(\mathbf{r}) = \rho_n(\mathbf{r})\,\exp\!\left(-\alpha\,\frac{f_n(\mathbf{r})}{dV}\right)
$$

where $f_n = \delta\Omega/\delta\rho \times dV$ is the discrete force and
$\alpha \in (0, 1)$ is the mixing parameter. The log-space form ensures
$\rho > 0$ at all iterations without explicit projection. Small $\alpha$
(e.g. $0.005$) gives stable but slow convergence; larger values risk
oscillation.

### Convergence criterion

The iteration converges when the root-mean-square force drops below a
tolerance:

$$
\left(\frac{1}{N}\sum_i f_i^2\right)^{1/2} < \varepsilon
$$

For constrained problems (fixed mass), the force residual may plateau at a
nonzero value corresponding to the Lagrange multiplier. In that case,
convergence is detected when $|\Omega_{n} - \Omega_{n-1}| < \varepsilon$.

### Fixed-mass constraint

For finding critical nuclei, the `fixed_mass_constraint` function enforces:

$$
\int \bigl[\rho(\mathbf{r}) - \rho_{\mathrm{bg}}\bigr]\, dV = N_{\mathrm{target}}
$$

by rescaling the excess density $\rho - \rho_{\mathrm{bg}}$ after each
Picard step. This turns the saddle point of the unconstrained grand
potential into a constrained stationary point.

## What the code does

1. Sets up a pure hard-sphere system (White Bear II FMT) on a
   $100 \times 20 \times 20$ grid at bulk packing fraction $\eta \approx 0.314$.

2. Initialises the density as uniform $\rho_{\mathrm{bulk}}$ with a small
   sinusoidal perturbation ($5\%$ amplitude).

3. Runs Picard iteration (mixing $\alpha = 0.005$, tolerance $10^{-8}$) to
   find the equilibrium density.

4. Verifies convergence:
   - The density returns to uniform (perturbation vanishes).
   - $\Omega/V = -P_{\mathrm{bulk}}$, confirming thermodynamic consistency.

## Build and run

```bash
make run-local
```
