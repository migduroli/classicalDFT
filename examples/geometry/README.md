# Geometry module

## Overview

The `dft_core::geometry` namespace provides mesh primitives at three
abstraction levels:

| Class | Role |
|-------|------|
| `Vertex` | A point in N-dimensional space |
| `Element` | A collection of vertices forming a mesh cell |
| `SquareBox` | A hypercube element (2D square, 3D cube) |
| `SUQMesh` (aliased as `Lattice`) | A structured uniform quad mesh |

These are organised by dimension under `two_dimensional` and
`three_dimensional` namespaces.

## Class hierarchy

```text
Vertex
Element
  └── SquareBox (abstract)
        ├── two_dimensional::SquareBox
        └── three_dimensional::SquareBox
Mesh (abstract)
  └── SUQMesh (abstract)
        ├── two_dimensional::SUQMesh  (= two_dimensional::Lattice)
        └── three_dimensional::SUQMesh (= three_dimensional::Lattice)
```

## Usage

```cpp
#include <classicaldft>
using namespace dft_core::geometry;

// Vertices
auto v1 = Vertex({0, 1, 2, 3});
auto v2 = Vertex({3, 4, 5, 6});
std::cout << "v1: " << v1 << std::endl;

// Elements with copy and move semantics
auto v_list = std::vector<Vertex>{v1, v2};
auto e1 = Element(v_list);           // copies
auto e2 = Element(std::move(v_list)); // moves (v_list emptied)

// 2D square-box elements
auto box = two_dimensional::SquareBox(0.25, {0, 0});

// 2D structured lattice: 1x1 domain, dx = 0.25
auto origin = std::vector<double>{0, 0};
auto lengths = std::vector<double>{1.0, 1.0};
auto lattice = two_dimensional::Lattice(0.25, lengths, origin);

// Python-style negative indexing
std::cout << lattice[{-1, -1}] << std::endl;
std::cout << "Volume: " << lattice.volume() << std::endl;
```

## Running

```bash
make run        # builds and runs inside Docker
make run-local  # builds and runs locally
```

## Plots

When built with `DFT_USE_MATPLOTLIB=ON` (default), plots are saved to `exports/`:

| File | Content |
|------|---------|
| `mesh_2d_suq.png` | SUQMesh 2D vertices (dx = 0.25) |
| `mesh_2d_uniform.png` | UniformMesh 2D with periodic boundary conditions |

![SUQMesh 2D](exports/mesh_2d_suq.png)
![UniformMesh 2D](exports/mesh_2d_uniform.png)

