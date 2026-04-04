# Fourier: FFTW3 RAII wrappers

## Purpose

This example demonstrates the FFTW3 wrappers (`FourierTransform` and
`FourierConvolution`) that underpin all DFT functional evaluations. Every FMT
weighted-density computation and every mean-field interaction integral is
performed as a convolution in Fourier space, so these wrappers are the core
computational engine of the library.

## Mathematical background

### The discrete Fourier transform (DFT)

For a real-valued function $f(\mathbf{r})$ sampled on an $N_x \times N_y \times N_z$
periodic grid, the forward (r2c) transform produces complex Fourier
coefficients:

$$
\hat{f}(\mathbf{k}) = \sum_{\mathbf{r}} f(\mathbf{r})\,e^{-2\pi i\,\mathbf{k}\cdot\mathbf{r}/\mathbf{N}}
$$

and the backward (c2r) transform recovers the real-space data:

$$
f(\mathbf{r}) = \frac{1}{N}\sum_{\mathbf{k}} \hat{f}(\mathbf{k})\,e^{+2\pi i\,\mathbf{k}\cdot\mathbf{r}/\mathbf{N}}
$$

where $N = N_x N_y N_z$. FFTW3's backward transform does **not** include the
$1/N$ factor, so the user must call `plan.scale(1.0 / N)` after `backward()`.

### Hermitian symmetry

Since $f(\mathbf{r}) \in \mathbb{R}$, the Fourier coefficients satisfy
$\hat{f}(-\mathbf{k}) = \hat{f}^*(\mathbf{k})$. FFTW3 exploits this by
storing only $N_x \times N_y \times (N_z/2 + 1)$ complex values (the r2c
layout), roughly halving the memory compared to a full complex transform.

The `FourierTransform` class exposes:

- `total()` — the number of real-space points $N$.
- `fourier_total()` — the number of stored complex coefficients
  $N_x \times N_y \times (N_z/2 + 1)$.

### Parseval's theorem

The total energy is preserved across the transform:

$$
\sum_{\mathbf{r}} |f(\mathbf{r})|^2 = \frac{1}{N}\sum_{\mathbf{k}} |\hat{f}(\mathbf{k})|^2
$$

The example verifies this by computing both sides and checking that their
ratio is $1.0$ (up to numerical precision).

### Cyclic convolution via FFT

The convolution of two periodic functions $g$ and $h$:

$$
(g \ast h)(\mathbf{r}) = \sum_{\mathbf{r}'} g(\mathbf{r}')\,h(\mathbf{r} - \mathbf{r}')
$$

is computed as a pointwise product in Fourier space:

$$
\widehat{g \ast h}(\mathbf{k}) = \hat{g}(\mathbf{k})\,\hat{h}(\mathbf{k})
$$

followed by an inverse transform. This reduces the cost from $O(N^2)$ to
$O(N\log N)$.

In the DFT library, this is how the FMT weighted densities are computed:
$n_\alpha(\mathbf{r}) = \rho \ast w_\alpha$ where $w_\alpha$ are the hard-sphere
weight functions. The `FourierConvolution` class wraps this pattern.

## What the code does

### 1. Round-trip test

Fills an $8^3$ grid with $\sin(x)$ replicated across the $y$-$z$ planes:

$$
f(i,j,k) = \sin\!\left(\frac{2\pi\,i}{N_x}\right) \quad \text{for all } j, k
$$

The forward transform produces Fourier coefficients concentrated at the
$k_x = \pm 1$ modes. After backward + scale, the maximum round-trip error
is at machine precision ($\sim 10^{-16}$).

### 2. Parseval's theorem verification

Computes the real-space energy $\sum |f|^2$ and the Fourier-space energy
$(1/N)\sum|\hat{f}|^2$ and prints their ratio (should be $\approx 1$).

### 3. FFT convolution: $\delta \ast 3 = 3$

Places a delta function in one input and a constant $3.0$ in the other.
The convolution theorem gives $(f \ast g)(\mathbf{r}) = 3.0$ everywhere,
and the code verifies the result range is $[3.0, 3.0]$.

## RAII design

Both `FourierTransform` and `FourierConvolution` manage FFTW3 plans and
aligned memory buffers via RAII. Plans are created in the constructor and
destroyed in the destructor. The `real()` and `fourier()` accessors return
`std::span` views into the internal buffers, avoiding copies.

## Build and run

```bash
make run-local
```
