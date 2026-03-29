# Compensated summation

## Overview

Naive floating-point summation accumulates rounding errors that grow linearly
with the number of terms. The `dft::numerics::arithmetic::summation`
namespace provides three compensated summation algorithms that keep the
worst-case error effectively independent of sequence length:

| Algorithm | Function | Error bound |
|-----------|----------|-------------|
| Kahan-Babuska | `kahan_babuska_sum` | O(epsilon) |
| Kahan-Babuska-Neumaier | `kahan_babuska_neumaier_sum` | Improved over KB |
| Kahan-Babuska-Klein | `kahan_babuska_klein_sum` | Second-order correction |

All three support incremental summation: pass the previous sum and error
state to continue accumulating across multiple batches.

The `CompensatedSum` class wraps these algorithms behind `+=` and `-=`
operators for ergonomic use.

## Usage

```cpp
#include <classicaldft>
using namespace dft::numerics::arithmetic;

// Free functions
auto x = std::vector<double>{1.0 + 1e-14, 2.5 + 1e-14, 3.0 + 1e-14};
auto [sum, err] = summation::kahan_babuska_neumaier_sum(x);

// Incremental continuation
auto y = std::vector<double>{4.0, 5.0};
auto [sum2, err2] = summation::kahan_babuska_neumaier_sum(y, sum, err);

// CompensatedSum class
summation::CompensatedSum cs;
cs += x;
cs += y;
std::cout << "sum = " << cs << std::endl;
```

## Expected output

```text
Test 1: Sum from scratch
Trivial sum [1]: 10.5
Kahan-Babuska sum [1]:10.5
Kahan-Babuska err [1] = -8.88178e-16
...

Test 3: Using CompensatedSum class
Testing the operator '+=' [1] = 10.5
Testing the operator '+=' [2] = 21.0014
```

## Running

```bash
make run   # builds and runs inside Docker
```

