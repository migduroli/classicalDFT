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

---

## Step-by-step code walkthrough

### Step 1: Round-trip test

An $8^3$ grid is filled with $\sin(x)$ replicated across the $y$-$z$ planes.
The forward and backward transforms are applied and the round-trip error
measured:

```cpp
auto plan = math::FourierTransform(shape);
arma::vec sin_3d = arma::repelem(sin_x, shape[1] * shape[2], 1);
auto real = plan.real();
std::copy(sin_3d.begin(), sin_3d.end(), real.begin());

plan.forward();
plan.backward();
plan.scale(1.0 / static_cast<double>(plan.total()));
```

The maximum error should be at machine precision ($\sim 10^{-16}$). The
Fourier coefficients are concentrated at $k_x = \pm 1$, confirming the
single-mode input.

### Step 2: Parseval's theorem verification

Real-space and Fourier-space energies are compared:

```cpp
double real_energy = arma::dot(roundtrip, roundtrip);
double fourier_energy = 0.0;
for (auto c : plan.fourier())
    fourier_energy += std::norm(c);
fourier_energy /= static_cast<double>(plan.total());
```

The ratio `fourier_energy / real_energy` should be $\approx 1$.

### Step 3: FFT convolution ($\delta * 3 = 3$)

The `FourierConvolution` class wraps the delta-constant convolution in a
single RAII object:

```cpp
auto conv = math::FourierConvolution(shape);
auto a = conv.input_a();
std::fill(a.begin(), a.end(), 0.0);
a[0] = 1.0;
auto b = conv.input_b();
std::fill(b.begin(), b.end(), 3.0);
conv.execute();
auto result = conv.result();
```

The result range must be $[3.0, 3.0]$ everywhere.

## RAII design

Both `FourierTransform` and `FourierConvolution` manage FFTW3 plans and
aligned memory buffers via RAII. Plans are created in the constructor and
destroyed in the destructor. The `real()` and `fourier()` accessors return
`std::span` views into the internal buffers, avoiding copies.

## Build and run

```bash
make run-local
```
