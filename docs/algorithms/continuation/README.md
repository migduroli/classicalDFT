# Continuation examples

This directory contains pseudo-arclength continuation examples for the
one-dimensional square-gradient model in two geometries:

- [`canonical/`](canonical/) solves the fixed-mass problem with Neumann walls.
  The walls pin a single interface and remove translational degeneracy.
- [`periodic/`](periodic/) solves the fixed-mass problem in a periodic box. Its
  translational symmetry requires a phase condition to select one
  representative of each translated profile.

Each example is self-contained, with its own `Makefile`, source files,
verification executable, and exported figures.
