# Fourier transforms

## Overview

The `dft_core::numerics::fourier` namespace provides RAII wrappers around FFTW3
for 3D real-to-complex transforms. The API is designed to feel like working with
a high-level library: create a plan, write into `std::span` views, and call
`forward()` / `backward()`.

| Class | Purpose |
|-------|---------|
| `GridShape` | Lightweight value type for 3D grid dimensions |
| `FourierTransform` | RAII wrapper for a pair of FFTW3 plans + aligned buffers |
| `FourierConvolution` | 3D cyclic convolution via pointwise Fourier multiplication |

## Usage

```cpp
#include <classicaldft>
using namespace dft_core::numerics;

// Define a grid
auto shape = GridShape{8, 8, 8};

// Create a plan (allocates FFTW buffers)
auto plan = fourier::FourierTransform(shape);

// Write data via std::span
auto real = plan.real();
real[0] = 1.0;

// Transform
plan.forward();
auto fk = plan.fourier();  // std::span<std::complex<double>>

// Round-trip
plan.backward();
plan.scale(1.0 / shape.total());

// Convolution
auto conv = fourier::FourierConvolution(shape);
auto a = conv.input_a();
auto b = conv.input_b();
// fill a and b...
conv.execute();
auto result = conv.result();
```

## Run

```bash
make run
```
