# Fourier: FFTW3 RAII wrappers

Demonstrates the FFTW3 wrappers for forward/backward FFT and convolution.

## What this example does

1. **Round-trip test**: fills an $8^3$ grid with $\sin(x)$ replicated across
   $y$-$z$ planes, performs a forward FFT, prints the first four Fourier
   coefficients, then backward-transforms and verifies the round-trip error
   is at machine precision.

2. **Parseval's theorem**: confirms that the total energy in real space equals
   the total energy in Fourier space (ratio $\approx 1$).

3. **FFT convolution**: convolves a delta function with a constant field
   ($\delta \ast 3 = 3$) using `FourierConvolution`, verifying the result is
   $3.0$ everywhere.

## Key API functions used

| Function | Purpose |
|----------|---------|
| `math::FourierTransform` | RAII forward/backward FFT |
| `FourierTransform::forward()` / `backward()` | execute transform |
| `FourierTransform::scale()` | normalise after backward |
| `math::FourierConvolution` | RAII cyclic convolution |

## Build and run

```bash
make run
```
