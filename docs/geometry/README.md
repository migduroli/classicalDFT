# Geometry: vertices, elements, and meshes

## Purpose

This example demonstrates the variant-based geometry module that provides the
computational grids underlying all DFT calculations. Every density profile
$\rho(\mathbf{r})$ lives on a uniform periodic mesh, so the geometry module
is a prerequisite for all physics docs.

## Mathematical background

### Uniform periodic mesh

A $d$-dimensional uniform mesh discretises a rectangular domain
$[0, L_1) \times [0, L_2) \times \cdots \times [0, L_d)$ with spacing
$\Delta x$ equal in all directions. The number of grid points along axis $k$
is $N_k = L_k / \Delta x$, giving a total of $\prod_k N_k$ grid points.

Grid point positions (using zero-indexed $i_k = 0, \ldots, N_k - 1$):

$$
\mathbf{r}_{i_1, i_2, \ldots, i_d} = (i_1\,\Delta x,\; i_2\,\Delta x,\; \ldots,\; i_d\,\Delta x)
$$

### Periodic wrapping

All meshes use periodic boundary conditions. A coordinate
$\mathbf{r} = (r_1, \ldots, r_d)$ is wrapped into the domain by:

$$
r_k \mapsto r_k - L_k\,\lfloor r_k / L_k \rfloor
$$

This maps any real-valued coordinate into $[0, L_k)$.

### Volume elements

Each grid point represents a cubic volume element (voxel) of volume:

- 2D: $V = (\Delta x)^2$
- 3D: $V = (\Delta x)^3$

These volumes appear in all DFT integrals as the quadrature weight:
$\int f(\mathbf{r})\,d\mathbf{r} \approx \Delta V \sum_i f(\mathbf{r}_i)$.

### Vertex type

The `Vertex` type is a thin wrapper around `std::vector<double>` that
supports element-wise arithmetic (`+`, `-`) and stream output. Vertices
are used both for grid-point positions and for general coordinate
manipulations.

---

## Key library types

| Type | Header | Role |
|------|--------|------|
| `Vertex` | `dft/geometry/vertex.hpp` | Coordinate vector with element-wise arithmetic |
| `Mesh` | `dft/geometry/mesh.hpp` | `std::variant<UniformMesh2D, UniformMesh3D>` with uniform spacing |
| `Element` | `dft/geometry/element.hpp` | `std::variant<SquareBox2D, SquareBox3D>` for grid cells |

---

## Step-by-step code walkthrough

### 1. Vertex arithmetic

Creates two 4D vertices and demonstrates addition and subtraction:

```cpp
auto v1 = geometry::Vertex{{0, 1, 2, 3}};
auto v2 = geometry::Vertex{{3, 4, 5, 6}};
// v1 + v2 = {3, 5, 7, 9}
// v2 - v1 = {3, 3, 3, 3}
```

### 2. 2D uniform mesh

Builds a $4 \times 4$ mesh with $\Delta x = 1.0$ (16 square elements).
Demonstrates periodic wrapping:

```cpp
auto mesh2d = geometry::uniform_mesh_2d(1.0, {4.0, 4.0}, {0.0, 0.0});
mesh2d.wrap({5.5, 7.0});   // → {1.5, 3.0}
mesh2d.wrap({-1.0, -0.5}); // → {3.0, 3.5}
```

### 3. 3D uniform mesh

Builds a $4 \times 4 \times 4$ mesh (64 cubic elements). Wraps a 3D
coordinate:

```cpp
auto mesh3d = geometry::uniform_mesh_3d(1.0, {4.0, 4.0, 4.0}, {0.0, 0.0, 0.0});
mesh3d.wrap({5.5, 7.0, 12.5}); // → {1.5, 3.0, 0.5}
```

### 4. Elements and volumes

Constructs `SquareBox2D` and `SquareBox3D` element objects and evaluates
their volume using method calls:

```cpp
auto sq = geometry::make_square_box_2d(1.0, {0.0, 0.0});
sq.volume(); // → 1.0

auto cb = geometry::make_square_box_3d(1.0, {0.0, 0.0, 0.0});
cb.volume(); // → 1.0
```

## Design notes

The `Mesh` type is a `std::variant<UniformMesh2D, UniformMesh3D>`, and
`Element` is a `std::variant<SquareBox2D, SquareBox3D>`. Functions like
`wrap()` and `volume()` use `std::visit` to dispatch on the active
alternative at runtime. This avoids virtual functions while keeping the
interface polymorphic.

## Build and run

```bash
make run-local
```
