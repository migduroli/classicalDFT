# Geometry: vertices, elements, and meshes

Demonstrates the variant-based geometry module for computational grids.

## What this example does

1. **Vertex arithmetic**: creates 4D vertices and demonstrates addition,
   subtraction, and stream output via `operator<<`.

2. **2D uniform mesh**: builds a $4 \times 4$ mesh with $\Delta x = 1.0$,
   prints element count and spacing, and demonstrates periodic wrapping for
   coordinates that fall outside the domain.

3. **3D uniform mesh**: builds a $4 \times 4 \times 4$ mesh and wraps a 3D
   coordinate.

4. **Elements**: constructs `SquareBox2D` and `SquareBox3D` elements and
   evaluates their volume via the variant-based `volume()` function.

## Key API functions used

| Function | Purpose |
|----------|---------|
| `geometry::uniform_mesh_2d()` | create 2D periodic mesh |
| `geometry::uniform_mesh_3d()` | create 3D periodic mesh |
| `geometry::wrap()` | periodic boundary wrapping |
| `geometry::make_square_box_2d()` / `3d()` | element factories |
| `geometry::volume()` | element volume |

## Build and run

```bash
make run
```
