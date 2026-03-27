# Thermodynamics

## Overview

The `dft_core::physics::thermodynamics` namespace provides hard-sphere fluid
models and equations of state for Lennard-Jones fluids.

| Class / namespace | Role |
|-------------------|------|
| `CarnahanStarling` | Carnahan-Starling hard-sphere model |
| `PercusYevick` | Percus-Yevick (virial / compressibility routes) |
| `transport` | Enskog transport coefficients (shear, bulk, thermal, sound damping) |
| `eos::IdealGas` | Ideal gas equation of state |
| `eos::PercusYevick` | Hard-sphere EOS via Percus-Yevick compressibility |
| `eos::LennardJonesJZG` | Johnson-Zollweg-Gubbins 32-parameter MBWR |
| `eos::LennardJonesMecke` | Mecke 32-term Lennard-Jones EOS |

## Usage

```cpp
#include <classicaldft>
using namespace dft_core::physics::thermodynamics;

// Hard-sphere pressure via Carnahan-Starling
CarnahanStarling cs;
double p = cs.pressure(0.3);  // P/(rho*kT) at eta=0.3

// Contact value (pair-correlation at contact)
double chi = cs.contact_value(0.3);

// Enskog transport
double eta_s = transport::shear_viscosity(0.5, chi);

// Lennard-Jones EOS
eos::LennardJonesJZG jzg(1.3);  // kT = 1.3
double p_lj = jzg.pressure(0.5);
```

## Running

```bash
make run   # builds and runs inside Docker
```
