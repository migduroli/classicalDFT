# Convolution: FFT-based cyclic convolution

## Purpose

This doc demonstrates the FFT convolution functions (`convolve` and
`back_convolve`) that implement all weighted-density and mean-field
integrals in the DFT library. Every FMT functional evaluation and every
mean-field interaction integral reduces to a Schur (element-wise) product
in Fourier space followed by an inverse transform.

## Mathematical background

### Cyclic convolution

For two periodic functions $f$ and $g$ sampled on an $N$-point grid, the
cyclic (circular) convolution is:

$$
(f \ast g)[n] = \sum_{m=0}^{N-1} f[m]\, g[n - m]
$$

where the index $n - m$ wraps around modulo $N$. By the convolution
theorem:

$$
\widehat{f \ast g}[k] = \hat{f}[k]\, \hat{g}[k]
$$

so the convolution can be computed as:

$$
f \ast g = \mathrm{IFFT}\bigl[\hat{f} \cdot \hat{g}\bigr]
$$

This reduces the cost from $O(N^2)$ to $O(N \log N)$.

### Forward convolution

The `convolve(weight_k, rho_k, shape)` function computes:

$$
n(\mathbf{r}) = \mathrm{IFFT}\bigl[\hat{w}(\mathbf{k})\, \hat{\rho}(\mathbf{k})\bigr]
$$

In DFT, this gives the weighted densities $n_\alpha = w_\alpha \ast \rho$
used by FMT, and the mean-field potential $\phi_{\mathrm{mf}} = w_{\mathrm{att}} \ast \rho$.

### Back-convolution (adjoint)

The `back_convolve(weight_k, derivative, shape)` function computes:

$$
\hat{b}(\mathbf{k}) = \hat{w}(\mathbf{k})\, \widehat{\frac{\partial\Phi}{\partial n_\alpha}}(\mathbf{k})
$$

This is needed for computing the functional derivative
$\delta F / \delta \rho$ by the chain rule through the weighted densities.
The key identity is the adjoint (transpose) property:

$$
\bigl\langle w \ast \rho,\, d \bigr\rangle = \bigl\langle \rho,\, w \ast d \bigr\rangle
$$

which ensures thermodynamic consistency of the force computation.

### Gaussian self-convolution

The convolution of two identical Gaussians
$g(x) = \exp\!\bigl(-x^2/(2\sigma^2)\bigr)$ is:

$$
(g \ast g)(x) = \sigma\sqrt{\pi}\;\exp\!\left(-\frac{x^2}{4\sigma^2}\right)
$$

i.e. a Gaussian with width $\sigma_{\mathrm{out}} = \sqrt{2}\,\sigma$ and
amplitude $\sigma\sqrt{\pi}$. This provides an analytical reference for
validating the numerical convolution.

## What the code does

1. **Delta convolution**: convolves $\delta(\mathbf{r})$ with a constant
   function $f = 3$. The result must be $3$ everywhere, verifying the
   identity property.

2. **Gaussian self-convolution**: convolves a Gaussian with itself on a
   1D periodic grid and compares the result with the analytical formula.
   Demonstrates the resolution-dependent accuracy of the FFT approach.

3. **Adjoint symmetry**: generates random fields $\rho$, $d$, and $w$,
   then verifies
   $\langle \mathrm{convolve}(w, \rho),\, d\rangle = \langle \rho,\, \mathrm{IFFT}[\mathrm{back\_convolve}(w, d)]\rangle$
   to machine precision. This identity is the mathematical foundation for
   correct force computation in heterogeneous DFT.

## Build and run

```bash
make run-local
```
