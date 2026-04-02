#include "dft.hpp"

#include <iostream>

using namespace dft;

int main() {

  // Vertex arithmetic.

  console::info("Vertex arithmetic");
  auto v1 = geometry::Vertex{{0, 1, 2, 3}};
  auto v2 = geometry::Vertex{{3, 4, 5, 6}};
  std::cout << "v1: " << v1 << "\n";
  std::cout << "v2: " << v2 << "\n";
  std::cout << "v1 + v2: " << v1 + v2 << "\n";
  std::cout << "v2 - v1: " << v2 - v1 << "\n";

  // 2D uniform mesh with periodic boundary conditions.

  console::info("UniformMesh2D (4x4, dx=1.0)");
  auto mesh2d = geometry::uniform_mesh_2d(1.0, {4.0, 4.0}, {0.0, 0.0});
  std::cout << "  Shape: [" << mesh2d.shape[0] << ", " << mesh2d.shape[1] << "]\n";
  std::cout << "  Elements: " << mesh2d.elements.size() << "\n";
  std::cout << "  Spacing: " << mesh2d.dx << "\n";

  geometry::Mesh m2d = mesh2d;
  auto w1 = geometry::wrap(m2d, geometry::Vertex{{1.5, 2.5}});
  auto w2 = geometry::wrap(m2d, geometry::Vertex{{5.5, 7.0}});
  auto w3 = geometry::wrap(m2d, geometry::Vertex{{-1.0, -0.5}});
  std::cout << "  wrap({1.5, 2.5}):   " << w1 << "\n";
  std::cout << "  wrap({5.5, 7.0}):   " << w2 << "\n";
  std::cout << "  wrap({-1.0, -0.5}): " << w3 << "\n";

  // 3D uniform mesh.

  console::info("UniformMesh3D (4x4x4, dx=1.0)");
  auto mesh3d = geometry::uniform_mesh_3d(1.0, {4.0, 4.0, 4.0}, {0.0, 0.0, 0.0});
  std::cout << "  Shape: [" << mesh3d.shape[0] << ", " << mesh3d.shape[1]
            << ", " << mesh3d.shape[2] << "]\n";
  std::cout << "  Elements: " << mesh3d.elements.size() << "\n";

  geometry::Mesh m3d = mesh3d;
  auto w4 = geometry::wrap(m3d, geometry::Vertex{{5.5, 7.0, 12.5}});
  std::cout << "  wrap({5.5, 7.0, 12.5}): " << w4 << "\n";

  // Element construction.

  console::info("Elements");
  auto sq = geometry::make_square_box_2d(1.0, {0.0, 0.0});
  std::cout << "  SquareBox2D volume: " << geometry::volume(sq) << "\n";

  auto cb = geometry::make_square_box_3d(1.0, {0.0, 0.0, 0.0});
  std::cout << "  SquareBox3D volume: " << geometry::volume(cb) << "\n";
}
